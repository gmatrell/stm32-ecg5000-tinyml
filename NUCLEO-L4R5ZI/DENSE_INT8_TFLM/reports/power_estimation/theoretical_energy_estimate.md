# Stima teorica dell energia per inferenza - NUCLEO-L4R5ZI

Data: 08 settembre 2026 09:40:42  
Tipo: stima teorica del solo microcontrollore, non misura fisica della scheda.

## Dati di partenza

- MCU: STM32L4R5ZI, die 470, Cortex-M4.
- Profilo STM32CubeMX: RUN, Flash dual-bank, Range1-Boost, CPU/AHB e APB1 a 120 MHz.
- Durata media dell inferenza DWT: 637.332 us. Seriale e protocollo sono esclusi.
- Periferiche abilitate nella finestra: GPIOG, LPUART1 e PWR. LPUART1 non trasferisce dati durante la misura.

I valori provengono dal database locale del Power Consumption Calculator STM32CubeMX 6.7: consumption_die_470.xml e die_470.xml. Il database espone 3,0 V e 3,6 V. Il risultato a 3,0 V e il valore diretto del profilo; quello a 3,25 V e un adeguamento lineare della tensione mantenendo la corrente del profilo.

## Calcolo

| Componente | Corrente stimata |
| --- | ---: |
| MCU RUN base | 17 mA |
| GPIOG a 120 MHz | 0.27 mA |
| LPUART1 a 120 MHz | 0.198 mA |
| PWR a 120 MHz | 0.06 mA |
| **Totale firmware stimato** | **17.528 mA** |

Profilo diretto a 3,0 V: P = 3 V x 17.528 mA = 52.584 mW  
E = 52.584 mW x 637.332 us / 1000 = 33.5135 uJ

Adeguamento a 3,25 V: E = 56.966 mW x 637.332 us / 1000 = 36.3063 uJ

Risultato principale: **circa 33.5 uJ per inferenza**. Energia MCU base, escluse le periferiche: **circa 32.5 uJ**.

## Limiti

- Non include ST-LINK, regolatori, LED, USB, connettori e altre perdite della scheda.
- Non include il tempo di trasferimento seriale PC-scheda.
- Il profilo di clock e una proxy del database. Una misura fisica richiede STM32CubeMonitor-Power e hardware dedicato.
