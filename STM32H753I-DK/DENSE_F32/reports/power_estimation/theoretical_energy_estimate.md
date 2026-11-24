# Stima energetica teorica MCU — STM32H573I-DK / DENSE_F32

## Esito

La stima di riferimento per una inferenza e' **16,0308 uJ**.

Non e' una misura fisica della STM32H573I-DK, ne' della sua alimentazione USB,
ST-LINK, LED, display o altri componenti della scheda. E' una stima del solo
MCU, basata sul datasheet, da usare esclusivamente assieme alle assunzioni qui
riportate.

## Configurazione verificata nel firmware

- STM32H573IIKxQ, Cortex-M33; VOS0.
- HSE digitale e PLL1: 25 MHz / PLLM 5 * PLLN 100 / PLLR 2 = 250 MHz;
  AHB, APB1, APB2 e APB3 senza prescaler.
- Codice eseguito dalla Flash interna; I-cache abilitata; D-cache non abilitata
  dal firmware.
- USART1 PA9/PA10 a 115200 baud e DWT CYCCNT abilitati.
- La finestra cronometrata comprende solo `ai_infer()` e vale 183 us medi su
  1.000 ECG; UART, checksum, argmax e trasmissione sono esclusi.

## Dato di corrente e calcolo

Il datasheet STM32H573xx, Tabella 30, riporta per `IDD(Run)` con codice e dati
processati da Flash, I-cache 1-way e prefetch ON, tutti i periferici disabilitati,
VOS0, fHCLK = 250 MHz, VDD = 3,0 V e TJ = 25 C:

- LDO tipico: 29,2 mA (massimo a 25 C: 34 mA).
- SMPS tipico: 15,9 mA.

Il firmware non seleziona esplicitamente LDO o SMPS e il datasheet non fornisce
un incremento in Run mode per USART1 inattiva durante la finestra DWT. La
stima primaria e' quindi il riferimento **LDO tipico**, non una corrente
misurata ne' il consumo totale del firmware.

`E_LDO_typ = 3,0 V * 29,2 mA * 183 us = 16,0308 uJ`

Riferimenti di sensibilita', con la stessa finestra:

- SMPS tipico: `3,0 V * 15,9 mA * 183 us = 8,7291 uJ`.
- LDO massimo a 25 C: `3,0 V * 34 mA * 183 us = 18,6660 uJ`.

## Limiti

La riga di datasheet e' il miglior riferimento pubblico disponibile, ma non
riproduce esattamente il carico X-CUBE-AI ne' quantifica USART1 e le correnti
dei pin PA9/PA10. Il valore 16,0308 uJ va dunque denominato **stima teorica MCU
di riferimento**, non energia della scheda. Una misura fisica richiede il punto
IDD (JP6) della DK e uno strumento di misura.

## Fonti

- STMicroelectronics, *STM32H573xx datasheet*, DS14121 Rev 5, Tabella 30:
  https://www.st.com/resource/en/datasheet/stm32h573mi.pdf
- STMicroelectronics, *Discovery kit with STM32H573II MCU*, UM3143 Rev 2,
  punto di misura IDD (JP6):
  https://www.st.com/resource/en/user_manual/um3143-discovery-kit-with-stm32h573ii-mcu-stmicroelectronics.pdf
