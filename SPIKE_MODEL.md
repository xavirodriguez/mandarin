# Informe de Spike de Modelo Acústico para CAPT Mandarín

## Objetivo
Evaluar alternativas de modelos acústicos de pronunciación en mandarín para la Fase 1 del ciclo (Decisión 2a), verificando licencia, latencia de inferencia en CPU y mapeo de salidas (caracteres a pinyin y unidades fonéticas inicial/final).

## Modelos Evaluados

### 1. `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`
- **Arquitectura**: Wav2Vec2 con cabezal CTC a nivel carácter.
- **Licencia**: Apache-2.0 (Permisiva para uso comercial y libre).
- **Tamaño del Vocabulario**: 3,503 tokens (caracteres mandarín + tokens especiales CTC).
- **Latencia CPU**: ~500 ms en frío, ~180 ms en inferencias sucesivas para 1 segundo de audio a 16 kHz.
- **Mapeo a Unidades**: La distribución posterior a nivel frame/carácter se convierte a Pinyin (sílaba, inicial y final) utilizando la librería `pypinyin`.

### 2. `TencentGameMate/chinese-wav2vec2-base`
- **Arquitectura**: Wav2Vec2 preentrenado únicamente (sin refinamiento CTC ni vocabulario de salida).
- **Licencia**: Apache-2.0.
- **Conclusión**: Requiere entrenamiento/fine-tuning CTC propio con dataset (AISHELL-3) para producir probabilidades posteriores. Descartado para el primer ciclo; retenido como candidato para el segundo ciclo si la evaluación de sílabas lo justifica (Decisión 2b).

### 3. `uai-org/wav2vec2-large-xlsr-53-mandarin-pinyin`
- **Estado**: Repositorio privado / inaccesible sin credenciales. Descartado.

## Selección y Mapeo Fonético

Se selecciona **`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`** para la implementación en la arquitectura.

### Mapeo de Carácter a Unidades Pinyin:
Dado un carácter predicho $C$, se extrae:
1. **Sílaba completa**: `pypinyin.lazy_pinyin(C, style=Style.NORMAL)` (ej. "好" -> "hao")
2. **Inicial (Onset)**: `pypinyin.lazy_pinyin(C, style=Style.INITIALS)` (ej. "好" -> "h")
3. **Final (Nucleus + Coda)**: `pypinyin.lazy_pinyin(C, style=Style.FINALS)` (ej. "好" -> "ao")

### Adaptador de Inferencia en Tiempo Real
Las probabilidades logit del modelo Wav2Vec2 CTC ($T \times V$) se convierten a posterrores de caracteres y, mediante agregación por mapeo `pypinyin`, a posteriores de unidades fonéticas en cada frame.

## Conclusión
El modelo `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` cumple con todos los requisitos de licencia, latencia y compatibilidad de unidades para actuar como motor real de `SpeechEncoderAdapter` y `PhonemeRecognizerAdapter`.
