#include "main.h"
#include "dense_int8.h"
#include "dense_int8_data.h"
#include "serial_protocol.h"

static UART_HandleTypeDef g_uart;
static ai_handle g_network = AI_HANDLE_NULL;
static uint8_t g_inference_timer_available = 0U;
AI_ALIGNED(4) static ai_u8 g_activations[AI_DENSE_INT8_DATA_ACTIVATIONS_SIZE];
AI_ALIGNED(4) static ai_float g_input[ECG5000_INPUT_FLOATS];
static ai_float g_output[AI_DENSE_INT8_OUT_1_SIZE];

static void SystemClock_Config(void);
static void ICACHE_Init(void);
static uint8_t InferenceTimer_Init(void);
static void USART1_Init(void);
static void uart_write(const char *text);
static int uart_receive_exact(uint8_t *buffer, uint32_t size);
static uint32_t fnv1a(const uint8_t *buffer, uint32_t size, uint32_t hash);
static uint32_t read_u32_le(const uint8_t *buffer);
static void write_u32_le(uint8_t *buffer, uint32_t value);
static void copy_bytes(uint8_t *destination, const uint8_t *source, uint32_t size);
static void zero_output(void);
static uint32_t argmax_class(const ai_float *values, uint32_t count);
static uint8_t ai_initialize(void);
static uint8_t ai_infer(void);
static uint32_t cycles_to_microseconds(uint32_t cycles);
static uint8_t receive_request(uint32_t *sequence);
static void send_response(uint32_t sequence, uint8_t status, uint8_t predicted_class,
                          uint32_t inference_microseconds);

int main(void)
{
  uint8_t ai_status;

  HAL_Init();
  SystemClock_Config();
  ICACHE_Init();
  g_inference_timer_available = InferenceTimer_Init();
  USART1_Init();
  ai_status = ai_initialize();
  uart_write("ECG5000_HW_READY board=STM32H573I-DK model=DENSE_INT8 uart=USART1 protocol=ECG1/RES2 request_bytes=572 response_bytes=40\r\n");

  while (1) {
    uint32_t sequence = 0U;
    uint32_t inference_microseconds = 0U;
    uint8_t status = receive_request(&sequence);
    uint8_t predicted_class = 0U;

    zero_output();
    if (status == ECG5000_STATUS_OK && ai_status != ECG5000_STATUS_OK) {
      status = ai_status;
    }
    if (status == ECG5000_STATUS_OK && g_inference_timer_available == 0U) {
      status = ECG5000_STATUS_TIMER_ERROR;
    }
    if (status == ECG5000_STATUS_OK) {
      uint32_t cycle_start = DWT->CYCCNT;
      status = ai_infer();
      uint32_t elapsed_cycles = DWT->CYCCNT - cycle_start;
      if (elapsed_cycles == 0U) {
        status = ECG5000_STATUS_TIMER_ERROR;
      } else {
        inference_microseconds = cycles_to_microseconds(elapsed_cycles);
      }
      if (status == ECG5000_STATUS_OK) {
        predicted_class = (uint8_t)(argmax_class(g_output, AI_DENSE_INT8_OUT_1_SIZE) + 1U);
      }
    }
    send_response(sequence, status, predicted_class, inference_microseconds);
  }
}

static uint8_t ai_initialize(void)
{
  const ai_handle activations[AI_DENSE_INT8_DATA_ACTIVATIONS_COUNT] = {
    AI_HANDLE_PTR(g_activations)
  };
  ai_error err = ai_dense_int8_create_and_init(&g_network, activations, NULL);
  return (err.type == AI_ERROR_NONE && err.code == AI_ERROR_CODE_NONE) ?
    ECG5000_STATUS_OK : ECG5000_STATUS_AI_INIT_ERROR;
}

static uint8_t ai_infer(void)
{
  ai_buffer *input = ai_dense_int8_inputs_get(g_network, NULL);
  ai_buffer *output = ai_dense_int8_outputs_get(g_network, NULL);
  input[0].data = AI_HANDLE_PTR(g_input);
  output[0].data = AI_HANDLE_PTR(g_output);
  return (ai_dense_int8_run(g_network, input, output) == 1) ?
    ECG5000_STATUS_OK : ECG5000_STATUS_AI_RUN_ERROR;
}

static uint32_t cycles_to_microseconds(uint32_t cycles)
{
  return (uint32_t)(((uint64_t)cycles * 1000000ULL + ((uint64_t)SystemCoreClock / 2ULL)) /
                    (uint64_t)SystemCoreClock);
}

static uint8_t receive_request(uint32_t *sequence)
{
  const uint8_t magic[4] = ECG5000_REQUEST_MAGIC;
  uint8_t sequence_bytes[4];
  uint8_t checksum_bytes[4];
  uint8_t byte = 0U;
  uint32_t matched = 0U;
  uint32_t expected_checksum;
  uint32_t received_checksum;
  uint32_t hash;

  while (matched < 4U) {
    if (uart_receive_exact(&byte, 1U) != 0) {
      return ECG5000_STATUS_CRC_ERROR;
    }
    if (byte == magic[matched]) {
      ++matched;
    } else {
      matched = (byte == magic[0]) ? 1U : 0U;
    }
  }
  if (uart_receive_exact(sequence_bytes, 4U) != 0 ||
      uart_receive_exact((uint8_t *)g_input, ECG5000_INPUT_BYTES) != 0 ||
      uart_receive_exact(checksum_bytes, 4U) != 0) {
    return ECG5000_STATUS_CRC_ERROR;
  }
  *sequence = read_u32_le(sequence_bytes);
  hash = fnv1a(sequence_bytes, 4U, 2166136261U);
  expected_checksum = fnv1a((const uint8_t *)g_input, ECG5000_INPUT_BYTES, hash);
  received_checksum = read_u32_le(checksum_bytes);
  return (expected_checksum == received_checksum) ? ECG5000_STATUS_OK : ECG5000_STATUS_CRC_ERROR;
}

static void send_response(uint32_t sequence, uint8_t status, uint8_t predicted_class,
                          uint32_t inference_microseconds)
{
  uint8_t response[ECG5000_RESPONSE_BYTES] = {0};
  uint32_t checksum;
  copy_bytes(response, (const uint8_t *)ECG5000_RESPONSE_MAGIC, 4U);
  write_u32_le(&response[4], sequence);
  response[8] = status;
  response[9] = predicted_class;
  write_u32_le(&response[12], inference_microseconds);
  copy_bytes(&response[16], (const uint8_t *)g_output, 20U);
  checksum = fnv1a(&response[4], 32U, 2166136261U);
  write_u32_le(&response[36], checksum);
  (void)HAL_UART_Transmit(&g_uart, response, ECG5000_RESPONSE_BYTES, HAL_MAX_DELAY);
}

static int uart_receive_exact(uint8_t *buffer, uint32_t size)
{
  return (HAL_UART_Receive(&g_uart, buffer, (uint16_t)size, HAL_MAX_DELAY) == HAL_OK) ? 0 : 1;
}

static uint32_t fnv1a(const uint8_t *buffer, uint32_t size, uint32_t hash)
{
  uint32_t i;
  for (i = 0U; i < size; ++i) {
    hash ^= buffer[i];
    hash *= 16777619U;
  }
  return hash;
}

static uint32_t read_u32_le(const uint8_t *buffer)
{
  return ((uint32_t)buffer[0]) |
         ((uint32_t)buffer[1] << 8U) |
         ((uint32_t)buffer[2] << 16U) |
         ((uint32_t)buffer[3] << 24U);
}

static void write_u32_le(uint8_t *buffer, uint32_t value)
{
  buffer[0] = (uint8_t)value;
  buffer[1] = (uint8_t)(value >> 8U);
  buffer[2] = (uint8_t)(value >> 16U);
  buffer[3] = (uint8_t)(value >> 24U);
}

static void copy_bytes(uint8_t *destination, const uint8_t *source, uint32_t size)
{
  uint32_t i;
  for (i = 0U; i < size; ++i) {
    destination[i] = source[i];
  }
}

static void zero_output(void)
{
  uint32_t i;
  for (i = 0U; i < AI_DENSE_INT8_OUT_1_SIZE; ++i) {
    g_output[i] = 0.0f;
  }
}

static uint32_t argmax_class(const ai_float *values, uint32_t count)
{
  uint32_t best = 0U;
  uint32_t i;
  for (i = 1U; i < count; ++i) {
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return best;
}

static void USART1_Init(void)
{
  GPIO_InitTypeDef gpio = {0};
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_USART1_CLK_ENABLE();

  gpio.Pin = GPIO_PIN_9 | GPIO_PIN_10;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_PULLUP;
  gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  gpio.Alternate = GPIO_AF7_USART1;
  HAL_GPIO_Init(GPIOA, &gpio);

  g_uart.Instance = USART1;
  g_uart.Init.BaudRate = 115200U;
  g_uart.Init.WordLength = UART_WORDLENGTH_8B;
  g_uart.Init.StopBits = UART_STOPBITS_1;
  g_uart.Init.Parity = UART_PARITY_NONE;
  g_uart.Init.Mode = UART_MODE_TX_RX;
  g_uart.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  g_uart.Init.OverSampling = UART_OVERSAMPLING_16;
  g_uart.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  g_uart.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  g_uart.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&g_uart) != HAL_OK) {
    Error_Handler();
  }
}

static void uart_write(const char *text)
{
  const char *cursor = text;
  while (*cursor != '\0') {
    (void)HAL_UART_Transmit(&g_uart, (uint8_t *)cursor, 1U, HAL_MAX_DELAY);
    ++cursor;
  }
}

static void SystemClock_Config(void)
{
  RCC_OscInitTypeDef oscillator = {0};
  RCC_ClkInitTypeDef clocks = {0};

  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);
  while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) { }
  oscillator.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  oscillator.HSEState = RCC_HSE_BYPASS_DIGITAL;
  oscillator.PLL.PLLState = RCC_PLL_ON;
  oscillator.PLL.PLLSource = RCC_PLL1_SOURCE_HSE;
  oscillator.PLL.PLLM = 5;
  oscillator.PLL.PLLN = 100;
  oscillator.PLL.PLLP = 2;
  oscillator.PLL.PLLQ = 2;
  oscillator.PLL.PLLR = 2;
  oscillator.PLL.PLLRGE = RCC_PLL1_VCIRANGE_2;
  oscillator.PLL.PLLVCOSEL = RCC_PLL1_VCORANGE_WIDE;
  oscillator.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&oscillator) != HAL_OK) { Error_Handler(); }

  clocks.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                     RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 |
                     RCC_CLOCKTYPE_PCLK3;
  clocks.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  clocks.AHBCLKDivider = RCC_SYSCLK_DIV1;
  clocks.APB1CLKDivider = RCC_HCLK_DIV1;
  clocks.APB2CLKDivider = RCC_HCLK_DIV1;
  clocks.APB3CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&clocks, FLASH_LATENCY_5) != HAL_OK) { Error_Handler(); }
  __HAL_FLASH_SET_PROGRAM_DELAY(FLASH_PROGRAMMING_DELAY_2);
  SystemCoreClockUpdate();
}

static void ICACHE_Init(void)
{
  if (HAL_ICACHE_Enable() != HAL_OK) { Error_Handler(); }
}

static uint8_t InferenceTimer_Init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0U;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  return (((DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) == 0U) &&
          ((DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U)) ? 1U : 0U;
}

void Error_Handler(void)
{
  __disable_irq();
  while (1) { }
}

void SysTick_Handler(void)
{
  HAL_IncTick();
}
