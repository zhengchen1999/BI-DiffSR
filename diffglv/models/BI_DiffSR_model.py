import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from diffglv.utils.base_model import BaseModel
from torch.nn import functional as F
import numpy as np
from diffusers import DDPMScheduler, DDIMScheduler

from diffglv.metrics.lpips import LPIPS

@MODEL_REGISTRY.register()
class BIDiffSRModel(BaseModel):
    """DiffIR model for stage two."""

    def __init__(self, opt):
        super(BIDiffSRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # diffusion
        self.set_new_noise_schedule(self.opt['beta_schedule'], self.device)

        # lipis
        self.lpips_opt = self.opt['val']['metrics'].get('lpips', None)
        if self.lpips_opt != None:
            self.lpips_metric = LPIPS(net="alex").to(self.device)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if self.cri_pix is None:
            raise ValueError('pixel loss is None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Network G: Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer)

    def set_new_noise_schedule(self, schedule_opt, device):
        scheduler_opt = self.opt['beta_schedule']
        scheduler_type = scheduler_opt.get('scheduler_type', None)
        _prediction_type = scheduler_opt.get('prediction_type', None)
        if scheduler_type == 'DDPM':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=schedule_opt['n_timestep'],
                                                beta_start=schedule_opt['linear_start'],
                                                beta_end=schedule_opt['linear_end'],
                                                beta_schedule=schedule_opt['schedule'])
        elif scheduler_type == 'DDIM':
            self.noise_scheduler = DDIMScheduler(num_train_timesteps=schedule_opt['n_timestep'],
                                                beta_start=schedule_opt['linear_start'],
                                                beta_end=schedule_opt['linear_end'],
                                                beta_schedule=schedule_opt['schedule'])
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
        
        if _prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=_prediction_type)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter, noise=None):
        self.optimizer.zero_grad()

        noise = torch.randn_like(self.gt).to(self.device)
        bsz = self.gt.shape[0]
        # Sample a random timestep for each image
        random_timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,), device=self.device)
        timesteps = random_timestep.repeat(bsz).long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_image = self.noise_scheduler.add_noise(self.gt, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(self.gt, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = self.gt
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        _timesteps = timesteps.unsqueeze(1).to(self.device)
        noise_pred = self.net_g(noisy_image, self.lq, _timesteps)
        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(noise_pred, target)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        scale = 1
        window_size = 8
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    
        if hasattr(self, 'net_g_ema'):
            print("TODO")
        else:
            self.net_g.eval()

            is_guidance = self.opt['beta_schedule'].get('is_guidance', False)

            if not is_guidance:
                # original conditional
                latents = torch.randn_like(img).to(self.device)

                self.noise_scheduler.set_timesteps(self.opt['beta_schedule']['num_inference_steps'])

                for t in self.noise_scheduler.timesteps:
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = latents
                    lq_image = img
                    _t = t.unsqueeze(0).unsqueeze(1).to(self.device)

                    latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)

                    # predict the noise residual
                    with torch.no_grad():
                        noise_pred = self.net_g(latent_model_input, lq_image, _t)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            else:
                # classifier-free guidance
                print("TODO")

            self.output = latents
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')

                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'lpips': continue
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                if self.lpips_opt != None:
                    sr_img = (img2tensor(metric_data['img']) / 255.0).unsqueeze(0).to(self.device)
                    hq_img = (img2tensor(metric_data['img2']) / 255.0).unsqueeze(0).to(self.device)
                    self.metric_results['lpips'] += self.lpips_metric(sr_img, hq_img, normalize=True, boundarypixels=self.lpips_opt['crop_border']).item()
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
