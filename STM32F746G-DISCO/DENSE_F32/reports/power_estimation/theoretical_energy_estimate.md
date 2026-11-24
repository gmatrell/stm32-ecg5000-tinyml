# Stima teorica dell'energia per inferenza - STM32F746

Data: 07 settembre 2026 10:06:54  
Tipo: stima teorica del solo microcontrollore, non misura fisica della scheda.

## Dati di partenza

- MCU: STM32F746NG, die 449.
- Alimentazione: 3.3 V.
- Temperatura: 25 C.
- Modalita': RUN, Scale1-High.
- Clock CPU/AHB: 216 MHz; APB2: 108 MHz.
- Memoria/fetch: SRAM/Flash con ART e cache L1 abilitate.
- Durata media dell'inferenza misurata dal DWT: 162.574 us.
- Periferiche del firmware abilitate nel periodo di inferenza: GPIOA, GPIOB, GPIOI e USART1. La UART e' abilitata ma non trasferisce dati nella finestra temporale misurata.

I valori di corrente provengono dai database locali del **Power Consumption Calculator** STM32CubeMX 6.7 incluso in STM32CubeIDE 1.11.2:

- consumption_die_449.xml: 108 mA per RUN, 3,3 V, 216 MHz, SRAM/Flash-ART/L1Cache.
- die_449.xml: contributi periferiche ConsumptionByMHz per GPIOA, GPIOB, GPIOI e USART1.

## Calcolo

| Componente | Corrente stimata |
| --- | ---: |
| MCU RUN base | 108 mA |
| GPIOA a 216 MHz | 0.4752 mA |
| GPIOB a 216 MHz | 0.4536 mA |
| GPIOI a 216 MHz | 0.4968 mA |
| USART1 a 108 MHz | 1.1124 mA |
| **Totale firmware stimato** | **110.538 mA** |

P = 3.3 V x 110.538 mA = 364.7754 mW

E = 364.7754 mW x 162.574 us / 1000 = 59.303 uJ

Risultato principale: **circa 59.3 uJ per inferenza**. Escludendo i contributi periferici, il valore base e' circa **57.94 uJ**.

## Limiti

- Non e' una misura della scheda STM32F746G-DISCO completa: non include display, ST-LINK, regolatori, LED, memoria esterna e perdite di alimentazione.
- Il consumo reale dipende da temperatura, tensione effettiva, dati elaborati, cache e attivita' dei bus.
- Per una misura sperimentale servirebbero STM32CubeMonitor-Power e hardware di misura, sincronizzati con un GPIO che delimiti l'inferenza.
