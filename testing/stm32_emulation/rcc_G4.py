# Minimal RCC + SysTick support for STM32G4 in Renode.
# This is enough for CubeMX SystemClock_Config + HAL_Delay + UART printf.

# ---- Offsets (subset) ----
CR        = 0x00
ICSCR     = 0x04
CFGR      = 0x08
PLLCFGR   = 0x0C
CIER      = 0x18
CIFR      = 0x1C
CICR      = 0x20
SYSCFG_CSR = 0x10   # SysTick CSR (fake)
SYSCFG_RVR = 0x14   # SysTick Reload Value
SYSCFG_CVR = 0x18   # SysTick Current Value

AHB1ENR   = 0x48
AHB2ENR   = 0x4C
AHB3ENR   = 0x50
APB1ENR1  = 0x58   # TIM2EN lives here
APB1ENR2  = 0x5C
APB2ENR   = 0x60
CCIPR     = 0x88
BDCR      = 0x90
CSR       = 0x94
CRRCR     = 0x98     # HSI48
CCIPR2    = 0xA8     # if present

# ---- Bits we care about ----
# CR
HSION   = 1 << 8
HSIRDY  = 1 << 10
HSEON   = 1 << 16
HSERDY  = 1 << 17
PLLON   = 1 << 24
PLLRDY  = 1 << 25

# CFGR
SW_MASK = 0b11
SWS_POS = 2

# CRRCR
HSI48ON  = 1 << 0
HSI48RDY = 1 << 1

# APB1ENR1
TIM2EN = 1 << 0     # CubeMX enables this if TIM2 is configured


# ---- State (register mirrors) ----
try:
    _rcc
except NameError:
    _rcc = {}

def _R(off):
    return _rcc.get(off, 0) & 0xFFFFFFFF

def _W(off, v):
    _rcc[off] = v & 0xFFFFFFFF

def _SET(off, mask, cond):
    v = _R(off)
    v = (v | mask) if cond else (v & ~mask)
    _W(off, v)


# ---- System Clock Switch Status ----
def _apply_sws_from_sw():
    cfgr = _R(CFGR)
    sw   = (cfgr & SW_MASK)
    cr   = _R(CR)

    if sw == 0 and (cr & HSIRDY):
        sws = 0
    elif sw == 1 and (cr & HSERDY):
        sws = 1
    elif sw == 2 and (cr & PLLRDY):
        sws = 2
    elif sw == 3:
        sws = 3
    else:
        sws = (cfgr >> SWS_POS) & 0x3

    cfgr = (cfgr & ~(0x3 << SWS_POS)) | (sws << SWS_POS)
    _W(CFGR, cfgr)


# ---- INIT ----
if request.isInit:
    # Reset-like defaults
    for o in (CR, ICSCR, CFGR, PLLCFGR, CIER, CIFR, CICR,
              AHB1ENR, AHB2ENR, AHB3ENR, APB1ENR1, APB1ENR2,
              APB2ENR, CCIPR, BDCR, CSR, CRRCR, CCIPR2,
              SYSCFG_CSR, SYSCFG_RVR, SYSCFG_CVR):
        _W(o, 0)

    # Enable HSI by default
    _SET(CR, HSION, True)
    _SET(CR, HSIRDY, True)
    _apply_sws_from_sw()

    # Fake SysTick defaults (HAL uses these)
    _W(SYSCFG_CSR, 0x5)     # ENABLE + CLKSOURCE
    _W(SYSCFG_RVR, 16000)   # pretend 1ms tick at 16 MHz
    _W(SYSCFG_CVR, 0)

    self.NoisyLog("RCC init: HSI ready, SysTick active (fake)")

# ---- NORMAL ACCESS ----
else:
    off = request.offset

    # ---- READ ----
    if request.isRead:
        request.value = _R(off)
        # self.NoisyLog(f"RCC read @ 0x{off:x} = 0x{request.value:x}")

    # ---- WRITE ----
    elif request.isWrite:
        val = request.value & 0xFFFFFFFF
        _W(off, val)
        # self.NoisyLog(f"RCC write @ 0x{off:x} = 0x{val:x}")

        # CR
        if off == CR:
            _SET(CR, HSIRDY, bool(val & HSION))
            _SET(CR, HSERDY, bool(val & HSEON))
            _SET(CR, PLLRDY, bool(val & PLLON))

        # CFGR
        elif off == CFGR:
            _apply_sws_from_sw()

        # HSI48
        elif off == CRRCR:
            _SET(CRRCR, HSI48RDY, bool(val & HSI48ON))

        # TIM2 clock detection
        elif off == APB1ENR1:
            if val & TIM2EN:
                self.NoisyLog("RCC: TIM2 clock enabled")

        # We already handled write, nothing else needed
        # All other registers simply store the value
