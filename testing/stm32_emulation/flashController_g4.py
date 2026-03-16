if request.isInit:
    Flash_ACR_LATENCY = 0x0
    Flash_ACR_LATENCY_value = 0x0
elif request.offset == Flash_ACR_LATENCY:
    if request.isRead:
        request.value = Flash_ACR_LATENCY_value
        self.NoisyLog("Read flash ACR latency: %d" % request.value)
    elif request.isWrite:
        Flash_ACR_LATENCY_value = request.value
        self.NoisyLog("Set flash ACR latency to: %d" % Flash_ACR_LATENCY_value)