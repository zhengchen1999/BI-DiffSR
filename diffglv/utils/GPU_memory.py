import pynvml
pynvml.nvmlInit()
gpuDeviceCount = pynvml.nvmlDeviceGetCount()
UNIT = 1024 * 1024
for i in range(gpuDeviceCount):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)#获取GPU i的handle，后续通过handle来处理

    memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息

    m_total = memoryInfo.total/UNIT
    m_used = memoryInfo.used/UNIT
    print('[%s][%s/%s]' % (i, m_used, m_total))
pynvml.nvmlShutdown()