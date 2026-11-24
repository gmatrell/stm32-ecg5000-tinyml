#include "main.h"
#include "serial_protocol.h"
#include "model_data.h"

#include <cmath>
#include <cstdint>
#include <new>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr uint32_t kOutputClasses = 5U;
constexpr size_t kTensorArenaBytes = 16384U;

UART_HandleTypeDef g_uart;
uint8_t g_inference_timer_available = 0U;
alignas(16) uint8_t g_tensor_arena[kTensorArenaBytes];
alignas(tflite::MicroInterpreter) uint8_t g_interpreter_storage[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output_tensor = nullptr;
alignas(4) float g_input_float[ECG5000_INPUT_FLOATS];
float g_output[kOutputClasses];
uint32_t g_arena_used = 0U;

void MPU_Config(void);
void CPU_CACHE_Enable(void);
void LED_Init(void);
void SystemClock_Config(void);
uint8_t InferenceTimer_Init(void);
void UART1_Init(void);
void uart_write(const char* text);
void tflm_log_callback(const char*) {}
int uart_receive_exact(uint8_t* buffer, uint32_t size);
uint32_t fnv1a(const uint8_t* buffer, uint32_t size, uint32_t hash);
uint32_t read_u32_le(const uint8_t* buffer);
void write_u32_le(uint8_t* buffer, uint32_t value);
void copy_bytes(uint8_t* destination, const uint8_t* source, uint32_t size);
void zero_output(void);
uint32_t argmax_class(const float* values, uint32_t count);
uint8_t tflm_initialize(void);
uint8_t tflm_infer(void);
uint32_t cycles_to_microseconds(uint32_t cycles);
uint8_t receive_request(uint32_t* sequence);
void send_response(uint32_t sequence, uint8_t status, uint8_t predicted_class,
                   uint32_t inference_microseconds);
}

int main(void) {
  MPU_Config();
  CPU_CACHE_Enable();
  HAL_Init();
  SystemClock_Config();
  g_inference_timer_available = InferenceTimer_Init();
  UART1_Init();
  LED_Init();
  const uint8_t runtime_status = tflm_initialize();
  uart_write("ECG5000_HW_READY board=STM32F746G-DISCO runtime=TFLM_reference protocol=ECG1/RES2 ");
  uart_write("request_bytes=572 response_bytes=40 tensor_arena_bytes=16384\r\n");

  while (true) {
    uint32_t sequence = 0U;
    uint32_t inference_microseconds = 0U;
    uint8_t status = receive_request(&sequence);
    uint8_t predicted_class = 0U;
    zero_output();
    if (status == ECG5000_STATUS_OK && runtime_status != ECG5000_STATUS_OK) status = runtime_status;
    if (status == ECG5000_STATUS_OK && g_inference_timer_available == 0U) status = ECG5000_STATUS_TIMER_ERROR;
    if (status == ECG5000_STATUS_OK) {
      const uint32_t start = DWT->CYCCNT;
      status = tflm_infer();
      const uint32_t elapsed = DWT->CYCCNT - start;
      if (elapsed == 0U) status = ECG5000_STATUS_TIMER_ERROR;
      else inference_microseconds = cycles_to_microseconds(elapsed);
      if (status == ECG5000_STATUS_OK) predicted_class = static_cast<uint8_t>(argmax_class(g_output, kOutputClasses) + 1U);
    }
    send_response(sequence, status, predicted_class, inference_microseconds);
  }
}

namespace {
uint8_t tflm_initialize(void) {
  RegisterDebugLogCallback(tflm_log_callback);
  const tflite::Model* model = tflite::GetModel(g_model_data);
  if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) return ECG5000_STATUS_AI_INIT_ERROR;
  static tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk || resolver.AddSoftmax() != kTfLiteOk) return ECG5000_STATUS_AI_INIT_ERROR;
  g_interpreter = new (g_interpreter_storage) tflite::MicroInterpreter(model, resolver, g_tensor_arena, kTensorArenaBytes);
  if (g_interpreter->AllocateTensors() != kTfLiteOk) return ECG5000_STATUS_AI_INIT_ERROR;
  g_input = g_interpreter->input(0);
  g_output_tensor = g_interpreter->output(0);
  if (g_input == nullptr || g_output_tensor == nullptr || g_input->type != kTfLiteFloat32 ||
      g_output_tensor->type != kTfLiteFloat32 || g_input->bytes != ECG5000_INPUT_FLOATS * sizeof(float) ||
      g_output_tensor->bytes != kOutputClasses * sizeof(float)) return ECG5000_STATUS_AI_INIT_ERROR;
  g_arena_used = static_cast<uint32_t>(g_interpreter->arena_used_bytes());
  return ECG5000_STATUS_OK;
}

uint8_t tflm_infer(void) {
  for (uint32_t i = 0U; i < ECG5000_INPUT_FLOATS; ++i)
    g_input->data.f[i] = g_input_float[i];
  if (g_interpreter->Invoke() != kTfLiteOk) return ECG5000_STATUS_AI_RUN_ERROR;
  for (uint32_t i = 0U; i < kOutputClasses; ++i)
    g_output[i] = g_output_tensor->data.f[i];
  return ECG5000_STATUS_OK;
}
}

namespace {
uint32_t cycles_to_microseconds(uint32_t cycles) {
  return static_cast<uint32_t>((static_cast<uint64_t>(cycles) * 1000000ULL +
      (static_cast<uint64_t>(SystemCoreClock) / 2ULL)) / static_cast<uint64_t>(SystemCoreClock));
}

uint8_t receive_request(uint32_t* sequence) {
  const uint8_t magic[4] = {'E', 'C', 'G', '1'};
  uint8_t sequence_bytes[4], checksum_bytes[4], byte = 0U;
  uint32_t matched = 0U;
  while (matched < 4U) {
    if (uart_receive_exact(&byte, 1U) != 0) return ECG5000_STATUS_CRC_ERROR;
    if (byte == magic[matched]) ++matched;
    else matched = (byte == magic[0]) ? 1U : 0U;
  }
  if (uart_receive_exact(sequence_bytes, 4U) != 0 ||
      uart_receive_exact(reinterpret_cast<uint8_t*>(g_input_float), ECG5000_INPUT_BYTES) != 0 ||
      uart_receive_exact(checksum_bytes, 4U) != 0) return ECG5000_STATUS_CRC_ERROR;
  *sequence = read_u32_le(sequence_bytes);
  uint32_t hash = fnv1a(sequence_bytes, 4U, 2166136261U);
  const uint32_t expected = fnv1a(reinterpret_cast<const uint8_t*>(g_input_float), ECG5000_INPUT_BYTES, hash);
  return (expected == read_u32_le(checksum_bytes)) ? ECG5000_STATUS_OK : ECG5000_STATUS_CRC_ERROR;
}

void send_response(uint32_t sequence, uint8_t status, uint8_t predicted_class, uint32_t inference_microseconds) {
  uint8_t response[ECG5000_RESPONSE_BYTES] = {0};
  copy_bytes(response, reinterpret_cast<const uint8_t*>(ECG5000_RESPONSE_MAGIC), 4U);
  write_u32_le(&response[4], sequence);
  response[8] = status; response[9] = predicted_class;
  write_u32_le(&response[12], inference_microseconds);
  copy_bytes(&response[16], reinterpret_cast<const uint8_t*>(g_output), 20U);
  write_u32_le(&response[36], fnv1a(&response[4], 32U, 2166136261U));
  (void)HAL_UART_Transmit(&g_uart, response, ECG5000_RESPONSE_BYTES, HAL_MAX_DELAY);
}

int uart_receive_exact(uint8_t* buffer, uint32_t size) {
  return (HAL_UART_Receive(&g_uart, buffer, static_cast<uint16_t>(size), HAL_MAX_DELAY) == HAL_OK) ? 0 : 1;
}
uint32_t fnv1a(const uint8_t* buffer, uint32_t size, uint32_t hash) {
  for (uint32_t i = 0U; i < size; ++i) { hash ^= buffer[i]; hash *= 16777619U; }
  return hash;
}
uint32_t read_u32_le(const uint8_t* buffer) {
  return static_cast<uint32_t>(buffer[0]) | (static_cast<uint32_t>(buffer[1]) << 8U) |
         (static_cast<uint32_t>(buffer[2]) << 16U) | (static_cast<uint32_t>(buffer[3]) << 24U);
}
void write_u32_le(uint8_t* buffer, uint32_t value) {
  buffer[0] = static_cast<uint8_t>(value); buffer[1] = static_cast<uint8_t>(value >> 8U);
  buffer[2] = static_cast<uint8_t>(value >> 16U); buffer[3] = static_cast<uint8_t>(value >> 24U);
}
void copy_bytes(uint8_t* destination, const uint8_t* source, uint32_t size) {
  for (uint32_t i = 0U; i < size; ++i) destination[i] = source[i];
}
void zero_output(void) { for (uint32_t i = 0U; i < kOutputClasses; ++i) g_output[i] = 0.0f; }
uint32_t argmax_class(const float* values, uint32_t count) {
  uint32_t best = 0U; for (uint32_t i = 1U; i < count; ++i) if (values[i] > values[best]) best = i; return best;
}

void UART1_Init(void)
{
  GPIO_InitTypeDef gpio = {};
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_USART1_CLK_ENABLE();

  gpio.Pin = GPIO_PIN_9;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_PULLUP;
  gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  gpio.Alternate = GPIO_AF7_USART1;
  HAL_GPIO_Init(GPIOA, &gpio);

  gpio.Pin = GPIO_PIN_7;
  HAL_GPIO_Init(GPIOB, &gpio);

  g_uart.Instance = USART1;
  g_uart.Init.BaudRate = 115200U;
  g_uart.Init.WordLength = UART_WORDLENGTH_8B;
  g_uart.Init.StopBits = UART_STOPBITS_1;
  g_uart.Init.Parity = UART_PARITY_NONE;
  g_uart.Init.Mode = UART_MODE_TX_RX;
  g_uart.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  g_uart.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&g_uart) != HAL_OK) {
    Error_Handler();
  }
}
void uart_write(const char* text) {
  while (*text != '\0') { (void)HAL_UART_Transmit(&g_uart, reinterpret_cast<uint8_t*>(const_cast<char*>(text)), 1U, HAL_MAX_DELAY); ++text; }
}

void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef clocks = {};
  RCC_OscInitTypeDef oscillator = {};
  oscillator.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  oscillator.HSEState = RCC_HSE_ON;
  oscillator.HSIState = RCC_HSI_OFF;
  oscillator.PLL.PLLState = RCC_PLL_ON;
  oscillator.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  oscillator.PLL.PLLM = 25U;
  oscillator.PLL.PLLN = 432U;
  oscillator.PLL.PLLP = RCC_PLLP_DIV2;
  oscillator.PLL.PLLQ = 9U;
  if (HAL_RCC_OscConfig(&oscillator) != HAL_OK) {
    Error_Handler();
  }
  if (HAL_PWREx_EnableOverDrive() != HAL_OK) {
    Error_Handler();
  }
  clocks.ClockType = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK |
                     RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  clocks.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  clocks.AHBCLKDivider = RCC_SYSCLK_DIV1;
  clocks.APB1CLKDivider = RCC_HCLK_DIV4;
  clocks.APB2CLKDivider = RCC_HCLK_DIV2;
  if (HAL_RCC_ClockConfig(&clocks, FLASH_LATENCY_7) != HAL_OK) {
    Error_Handler();
  }
}
uint8_t InferenceTimer_Init(void)
{
  /* DWT CYCCNT conta i cicli CPU della sola inferenza. */
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->LAR = 0xC5ACCE55U;
  DWT->CYCCNT = 0U;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  return (((DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) == 0U) &&
          ((DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U)) ? 1U : 0U;
}
}

namespace {
void MPU_Config(void)
{
  MPU_Region_InitTypeDef region = {};
  HAL_MPU_Disable();
  region.Enable = MPU_REGION_ENABLE;
  region.BaseAddress = 0x00U;
  region.Size = MPU_REGION_SIZE_4GB;
  region.AccessPermission = MPU_REGION_NO_ACCESS;
  region.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
  region.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  region.IsShareable = MPU_ACCESS_SHAREABLE;
  region.Number = MPU_REGION_NUMBER0;
  region.TypeExtField = MPU_TEX_LEVEL0;
  region.SubRegionDisable = 0x87U;
  region.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  HAL_MPU_ConfigRegion(&region);
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}
void CPU_CACHE_Enable(void)
{
  SCB_EnableICache();
  SCB_EnableDCache();
}
void LED_Init(void)
{
  GPIO_InitTypeDef gpio = {};
  __HAL_RCC_GPIOI_CLK_ENABLE();
  gpio.Pin = GPIO_PIN_1;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOI, &gpio);
  HAL_GPIO_WritePin(GPIOI, GPIO_PIN_1, GPIO_PIN_RESET);
}
}

extern "C" void Error_Handler(void) { __disable_irq(); while (true) {} }
