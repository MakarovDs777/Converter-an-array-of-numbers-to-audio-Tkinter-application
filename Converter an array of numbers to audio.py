import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
import struct
from pydub import AudioSegment

# Новые импорты для функции "Случайный звук из файла"
import threading
import random
import numpy as np
import sounddevice as sd

# Параметры по умолчанию (используются в режиме "legacy")
DEFAULT_SAMPLE_RATE = 44100  # Гц
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2     # bytes (16-bit)


# Глобальные переменные для управления циклом случайного воспроизведения
random_loop_thread = None
random_loop_stop_event = None


def parse_number_array(text):
    """
    Разбирает текстовый ввод чисел. Поддерживает два режима:
    1) Современный (рекомендуемый): первая строка может быть метаданной, начинающейся с "#META",
       например: "#META mode=bytes sample_rate=44100 channels=1 sample_width=2"
       В этом режиме числа интерпретируются как байты 0..255 (raw PCM data).
    2) Legacy-режим (по умолчанию): список чисел 0..255, где каждое число соответствует ОДНОМУ сэмплу
       и применяется прежнее масштабирование (не обязательно обратимое).

    Возвращает кортеж (numbers: list[int], meta: dict|None).
    """
    if not text:
        raise ValueError("Пустой ввод.")
    cleaned = text.strip()
    lines = cleaned.splitlines()
    meta = None
    if lines and lines[0].strip().startswith('#META'):
        header = lines[0].strip()[5:].strip()
        meta = {}
        for part in header.split():
            if '=' in part:
                k, v = part.split('=', 1)
                # Попробуем привести числовые значения к int
                if re.fullmatch(r"\d+", v):
                    meta[k] = int(v)
                else:
                    meta[k] = v
        nums_text = '\n'.join(lines[1:])
    else:
        nums_text = cleaned

    nums = re.findall(r"-?\d+", nums_text)
    if not nums:
        raise ValueError("Не найдено чисел в вводе или файле.")
    ints = [int(s) for s in nums]
    # Проверяем диапазон 0..255 для всех чисел
    for n in ints:
        if n < 0 or n > 255:
            raise ValueError(f"Число вне диапазона 0..255: {n}")
    return ints, meta


def numbers_to_16bit_pcm(numbers):
    """
    Преобразует список чисел 0..255 в signed 16-bit little-endian PCM
    (legacy-режим, где каждое число — один сэмпл).
    Масштабирование: 0 -> -32768, 128 -> ~0, 255 -> 32767
    """
    pcm_bytes = bytearray()
    for x in numbers:
        val = int(((x - 128) / 127.0) * 32767.0)
        if val < -32768:
            val = -32768
        if val > 32767:
            val = 32767
        pcm_bytes.extend(struct.pack('<h', val))
    return bytes(pcm_bytes)


def _bytes_to_float32(raw_bytes, sample_width, channels):
    """
    Преобразует сырые PCM-байты в float32 в диапазоне [-1.0, 1.0] для sounddevice.
    Поддерживаем типичные 8-bit unsigned и 16-bit signed.
    Возвращает numpy.ndarray dtype=float32 с формой (n_frames,) или (n_frames, channels).
    """
    if len(raw_bytes) == 0:
        return np.zeros((0,), dtype=np.float32)

    if sample_width == 1:
        # 8-bit PCM обычно unsigned: 0..255 -> -1..1
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sample_width == 2:
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        arr = arr / 32768.0
    else:
        # fallback: попробуем интерпретировать как int16
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        arr = arr / 32768.0

    if channels > 1:
        try:
            arr = arr.reshape(-1, channels)
        except Exception:
            # если reshape не удался, попытаемся добавить ось и дублировать
            arr = arr.reshape(-1, 1)
            if arr.shape[1] != channels:
                # продублируем канал в случае несовпадения
                arr = np.tile(arr, (1, channels))
    return arr.astype(np.float32)


def _loop_play_random_from_file(numbers, meta, stop_event):
    """
    Открывает sounddevice OutputStream один раз и в цикле пишет в него сгенерированные
    случайные непрерывные фрагменты (минимизируем паузы).
    """
    if not numbers:
        return

    # Параметры аудио
    frame_rate = DEFAULT_SAMPLE_RATE
    channels = DEFAULT_CHANNELS
    sample_width = DEFAULT_SAMPLE_WIDTH
    if meta:
        try:
            if 'sample_rate' in meta:
                frame_rate = int(meta['sample_rate'])
            if 'channels' in meta:
                channels = int(meta['channels'])
            if 'sample_width' in meta:
                sample_width = int(meta['sample_width'])
        except Exception:
            pass

    # Параметры генерации
    min_chunk_ms = 100   # минимальный кусок в ms
    max_chunk_ms = 2000  # максимальный кусок в ms
    play_buffer_ms = 5000  # длина буфера для одной записи в поток (ms)

    bytes_per_sec = frame_rate * channels * sample_width

    byte_array = bytes(numbers)
    nbytes = len(byte_array)
    if nbytes == 0:
        return

    # Открываем единый поток для вывода (dtype float32)
    try:
        with sd.OutputStream(samplerate=frame_rate, channels=channels, dtype='float32', latency='low') as stream:
            while not stop_event.is_set():
                # формируем один большой непрерывный байтовый буфер
                buffer_bytes = bytearray()
                buffer_ms_acc = 0
                while buffer_ms_acc < play_buffer_ms and not stop_event.is_set():
                    chunk_ms = random.randint(min_chunk_ms, max_chunk_ms)
                    needed_bytes = max(1, int(bytes_per_sec * (chunk_ms / 1000.0)))
                    if needed_bytes <= nbytes:
                        start = random.randint(0, nbytes - needed_bytes)
                        sel = byte_array[start:start + needed_bytes]
                    else:
                        reps = (needed_bytes + nbytes - 1) // nbytes
                        sel = (byte_array * reps)[:needed_bytes]
                    buffer_bytes.extend(sel)
                    # приближённо пересчитываем длину в ms
                    buffer_ms_acc = int((len(buffer_bytes) / bytes_per_sec) * 1000)

                if len(buffer_bytes) == 0:
                    continue

                # Конвертируем в float32 и пишем в поток — запись происходит непрерывно в открытый поток
                audio_np = _bytes_to_float32(buffer_bytes, sample_width, channels)
                try:
                    stream.write(audio_np)
                except Exception:
                    # если запись упала — даём короткий таймаут и пробуем снова
                    if stop_event.wait(0.05):
                        break
                    continue
    except Exception as e:
        # При ошибке открытия аудиоустройства — покажем пользователю (GUI-поток не блокируем здесь)
        try:
            messagebox.showerror("Ошибка аудио", f"Не удалось открыть аудиопоток: {e}")
        except Exception:
            pass
        return


def create_mp3_from_numbers(numbers, meta=None, duration_sec=None, output_name="evp_audio.mp3"):
    """
    Создаёт mp3 на Рабочем столе из списка чисел (0..255).
    Если meta указывает mode=bytes, то числа интерпретируются как байты raw PCM и
    восстанавливаются исходные параметры sample_rate/channels/sample_width (lossless при сохранении).
    Если meta отсутствует — используется legacy поведение (одно число = один сэмпл).

    Если duration_sec указана, результат будет обрезан или дополнен тишиной до этой длины.
    Возвращает (output_path, duration_seconds).
    """
    if not numbers:
        raise ValueError("Нет чисел для создания аудио.")

    if meta and str(meta.get('mode', '')).lower() == 'bytes':
        # Интерпретируем как raw bytes
        sample_width = int(meta.get('sample_width', DEFAULT_SAMPLE_WIDTH))
        frame_rate = int(meta.get('sample_rate', DEFAULT_SAMPLE_RATE))
        channels = int(meta.get('channels', DEFAULT_CHANNELS))
        # Сформируем байтовую последовательность
        raw = bytes(numbers)
        # Если raw длина не кратна sample_width * channels, AudioSegment всё равно попытается прочитать
        audio = AudioSegment(data=raw, sample_width=sample_width, frame_rate=frame_rate, channels=channels)
    else:
        # Legacy: каждое число — один сэмпл (масштабируем)
        pcm = numbers_to_16bit_pcm(numbers)
        audio = AudioSegment(data=pcm, sample_width=DEFAULT_SAMPLE_WIDTH, frame_rate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS)

    # Подгонка длительности
    if duration_sec is not None:
        if duration_sec <= 0:
            raise ValueError("Длительность должна быть положительным числом.")
        target_ms = int(duration_sec * 1000)
        if len(audio) > target_ms:
            audio = audio[:target_ms]
        elif len(audio) < target_ms:
            silence = AudioSegment.silent(duration=(target_ms - len(audio)), frame_rate=audio.frame_rate)
            audio = audio + silence
        actual_duration = duration_sec
    else:
        actual_duration = len(audio) / 1000.0

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    output_path = os.path.join(desktop, output_name)
    audio.export(output_path, format="mp3")
    return output_path, actual_duration


def audio_to_numbers_and_save(file_path):
    """
    Загружает аудио-файл (любого формата, поддерживаемого pydub/ffmpeg),
    и сохраняет текстовый файл на рабочем столе в формате:

    #META mode=bytes sample_rate=44100 channels=1 sample_width=2
    12 34 255 0  ...  # все байты raw PCM (little-endian)

    Таким образом, при обратном преобразовании можно восстановить исходный звук без потерь.
    Возвращает путь к сохранённому текстовому файлу и количество сэмплов (фактических аудио-семплов).
    """
    if not file_path:
        raise ValueError("Не выбран файл.")
    audio = AudioSegment.from_file(file_path)
    # Получаем текущие параметры аудио
    frame_rate = audio.frame_rate
    channels = audio.channels
    sample_width = audio.sample_width

    raw = audio.raw_data  # bytes
    nums = list(raw)  # ints 0..255

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_name = f"{base}_numbers.txt"
    out_path = os.path.join(desktop, out_name)

    header = f"#META mode=bytes sample_rate={frame_rate} channels={channels} sample_width={sample_width}\n"
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(header)
            # Запишем числа в одну строку для компактности
            f.write(' '.join(str(n) for n in nums))
    except Exception as e:
        raise IOError(f"Не удалось записать файл чисел: {e}")

    # Количество сэмплов (в зависимости от sample_width и channels)
    samples = len(raw) // (sample_width * max(1, channels))
    return out_path, samples


# --- GUI ---
root = tk.Tk()
root.title("Конвертер чисел <-> MP3/Audio")
root.geometry("980x620")

# Инструкция
tk.Label(root, text=("Введите массив чисел (0..255) или загрузите текстовый файл со значениями.\n"
                       "Если файл был получен через кнопку 'Загрузить аудио', в нём будет метаданные в первой строке '#META ...' —\n"
                       "это позволяет восстановить исходное аудио без потерь (mode=bytes).\n"
                       "Иначе используется legacy-режим (одно число = один сэмпл) — он не всегда обратим.)")).pack(pady=(12,6))

# Многострочное поле ввода
text_input = tk.Text(root, height=16, width=120)
text_input.pack(pady=6)
text_input.insert("1.0", "128 128 128 255 0 64 192")  # пример

info_label = tk.Label(root, text=("Дефолт: 44100 Hz, моно, 16-bit. При использовании mode=bytes будут применены параметры,\n"
                                  "указанные в метаданных (#META).") )
info_label.pack(pady=4)

# Поле для ввода желаемой длины
len_frame = tk.Frame(root)
len_frame.pack(pady=6)

tk.Label(len_frame, text="Желаемая длина (секунд, опционально; если пусто — длина = исходная/derived):").pack(side=tk.LEFT)
entry_duration = tk.Entry(len_frame, width=12)
entry_duration.pack(side=tk.LEFT, padx=8)
entry_duration.insert(0, "")  # по умолчанию пусто

# Кнопки
button_frame = tk.Frame(root)
button_frame.pack(pady=12)


def parse_duration_field():
    s = entry_duration.get().strip()
    if not s:
        return None
    try:
        val = float(s)
        if val <= 0:
            raise ValueError
        return val
    except Exception:
        raise ValueError("Некорректная длина: введите положительное число секунд или оставьте поле пустым.")


def on_create_from_text():
    raw_text = text_input.get("1.0", tk.END).strip()
    try:
        numbers, meta = parse_number_array(raw_text)
    except ValueError as e:
        messagebox.showerror("Ошибка ввода", str(e))
        return
    try:
        duration = parse_duration_field()
    except ValueError as e:
        messagebox.showerror("Ошибка длины", str(e))
        return
    try:
        out_path, duration_res = create_mp3_from_numbers(numbers, meta=meta, duration_sec=duration)
    except Exception as e:
        messagebox.showerror("Ошибка при создании аудио", str(e))
        return
    desc = f"MP3 сохранён:\n{out_path}\nДлительность: {duration_res:.3f} с"
    if meta:
        desc += f"\nИспользованы параметры из META: {meta}"
    messagebox.showinfo("Готово", desc)


def on_load_file():
    file_path = filedialog.askopenfilename(title="Выберите текстовый файл",
                                           filetypes=[("Text files", ".txt .csv .log .dat"), ("All files", "*")])
    if not file_path:
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")
        return
    text_input.delete('1.0', tk.END)
    text_input.insert('1.0', data)
    messagebox.showinfo("Файл загружен", f"Файл {os.path.basename(file_path)} загружен в поле ввода.")


def on_load_file_and_create():
    file_path = filedialog.askopenfilename(title="Выберите текстовый файл",
                                           filetypes=[("Text files", ".txt .csv .log .dat"), ("All files", "*")])
    if not file_path:
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")
        return
    try:
        numbers, meta = parse_number_array(data)
    except ValueError as e:
        messagebox.showerror("Ошибка парсинга", str(e))
        return
    try:
        duration = parse_duration_field()
    except ValueError as e:
        messagebox.showerror("Ошибка длины", str(e))
        return
    try:
        out_path, duration_res = create_mp3_from_numbers(numbers, meta=meta, duration_sec=duration)
    except Exception as e:
        messagebox.showerror("Ошибка при создании аудио", str(e))
        return
    text_input.delete('1.0', tk.END)
    text_input.insert('1.0', data)
    desc = f"MP3 сохранён:\n{out_path}\nДлительность: {duration_res:.3f} с"
    if meta:
        desc += f"\nИспользованы параметры из META: {meta}"
    messagebox.showinfo("Готово", desc)


def on_load_audio_and_convert():
    file_path = filedialog.askopenfilename(title="Выберите аудиофайл",
                                           filetypes=[("Audio files", "*.mp3 *.mp4 *.m4a *.wav *.flac *.aac *.ogg"), ("All files", "*")])
    if not file_path:
        return
    try:
        out_path, samples = audio_to_numbers_and_save(file_path)
    except Exception as e:
        messagebox.showerror("Ошибка при обработке аудио", str(e))
        return
    # Показать начало файла чисел в поле ввода для предварительного просмотра
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            data = f.read()
        text_input.delete('1.0', tk.END)
        text_input.insert('1.0', data[:10000] + ("..." if len(data) > 10000 else ""))
    except Exception:
        pass
    # Примерная длительность
    # Получим sample_rate из заголовка файла
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            first = f.readline()
        sr = DEFAULT_SAMPLE_RATE
        m = re.search(r"sample_rate=(\d+)", first)
        if m:
            sr = int(m.group(1))
        duration = samples / sr
    except Exception:
        duration = samples / DEFAULT_SAMPLE_RATE
    messagebox.showinfo("Готово", f"Файл чисел сохранён на рабочем столе:\n{out_path}\nСэмплов: {samples}\nПримерная длительность: {duration:.3f} с")


def on_random_sound_from_file_toggle():
    """
    Обработчик кнопки "Случайный звук из файла (Старт/Стоп)".
    Запускает/останавливает фоновый поток, который непрерывно проигрывает случайные фрагменты.
    """
    global random_loop_thread, random_loop_stop_event, random_file_btn

    # Если цикл уже запущен — остановим
    if random_loop_thread and random_loop_thread.is_alive():
        random_loop_stop_event.set()
        random_loop_thread.join(timeout=2.0)
        random_loop_thread = None
        random_loop_stop_event = None
        messagebox.showinfo("Стоп", "Бесконечный цикл случайных звуков остановлен.")
        random_file_btn.config(text="Случайный звук из файла (Старт)")
        return

    # Выбрать файл
    file_path = filedialog.askopenfilename(title="Выберите текстовый файл с числами",
                                           filetypes=[("Text files", ".txt .csv .log .dat"), ("All files", "*")])
    if not file_path:
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")
        return

    try:
        numbers, meta = parse_number_array(data)
    except ValueError as e:
        messagebox.showerror("Ошибка парсинга", str(e))
        return

    # Готово — запускаем цикл из данных файла
    stop_event = threading.Event()
    random_loop_stop_event = stop_event
    thread = threading.Thread(target=_loop_play_random_from_file, args=(numbers, meta, stop_event), daemon=True)
    random_loop_thread = thread
    thread.start()
    messagebox.showinfo("Старт", "Запущен бесконечный цикл случайных звуков из выбранного файла (нажатие кнопки остановит).")
    random_file_btn.config(text="Случайный звук из файла (Стоп)")


create_btn = tk.Button(button_frame, text="Создать MP3 из поля ввода", command=on_create_from_text, width=30)
create_btn.grid(row=0, column=0, padx=8, pady=4)

load_btn = tk.Button(button_frame, text="Загрузить файл в поле ввода", command=on_load_file, width=30)
load_btn.grid(row=0, column=1, padx=8, pady=4)

load_create_btn = tk.Button(button_frame, text="Загрузить файл и создать MP3", command=on_load_file_and_create, width=30)
load_create_btn.grid(row=0, column=2, padx=8, pady=4)

# НОВАЯ КНОПКА: загрузить аудио и разложить в числа (lossless)
audio_btn = tk.Button(button_frame, text="Загрузить аудио и сохранить числа (lossless)", command=on_load_audio_and_convert, width=44)
audio_btn.grid(row=1, column=0, columnspan=2, padx=8, pady=6)

# НОВАЯ КНОПКА: Случайный звук из файла (Старт/Стоп)
random_file_btn = tk.Button(button_frame, text="Случайный звук из файла (Старт)", command=on_random_sound_from_file_toggle, width=44)
# Помещаем в свободную ячейку (ряд 1, колонка 2) рядом с audio_btn
random_file_btn.grid(row=1, column=2, padx=8, pady=6)

quit_btn = tk.Button(root, text="Выйти", command=root.quit, width=12)
quit_btn.pack(pady=12)

root.mainloop()
