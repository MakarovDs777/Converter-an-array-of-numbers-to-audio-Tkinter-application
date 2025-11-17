import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
import struct
from pydub import AudioSegment

# Параметры аудио
SAMPLE_RATE = 44100  # Гц
CHANNELS = 1
SAMPLE_WIDTH = 2     # bytes (16-bit)


def parse_number_array(text):
    """
    Разбирает строку со числами. Поддерживает форматы:
    - "1 2 3 4"
    - "1,2,3"
    - "[1, 2, 3]"
    - любые комбинации пробелов, запятых и переводов строки.

    Возвращает list[int]. Проверяет диапазон 0..255.
    """
    if not text:
        raise ValueError("Пустой ввод.")
    cleaned = text.strip()
    # Удалим квадратные скобки, если они есть
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    # Найдём все целые числа (включая отрицательные, если вдруг)
    nums = re.findall(r"-?\d+", cleaned)
    if not nums:
        raise ValueError("Не найдено чисел в вводе или файле.")
    ints = []
    for s in nums:
        n = int(s)
        # Проверяем допустимый диапазон (0..255)
        if n < 0 or n > 255:
            raise ValueError(f"Число вне диапазона 0..255: {n}")
        ints.append(n)
    return ints


def numbers_to_16bit_pcm(numbers):
    """
    Преобразует список чисел 0..255 в signed 16-bit little-endian PCM.
    Масштабирование: 0 -> -32768, 128 -> ~0, 255 -> 32767
    Каждое число соответствует одному сэмплу (моно).
    """
    pcm_bytes = bytearray()
    for x in numbers:
        # масштабируем в диапазон -32768..32767
        val = int(((x - 128) / 127.0) * 32767.0)
        if val < -32768:
            val = -32768
        if val > 32767:
            val = 32767
        pcm_bytes.extend(struct.pack('<h', val))
    return bytes(pcm_bytes)


def create_mp3_from_numbers(numbers, duration_sec=None, output_name="evp_audio.mp3"):
    """
    Создаёт mp3 на Рабочем столе из списка чисел (0..255).
    Если duration_sec указана, результат будет обрезан или дополнен тишиной до этой длины.
    Возвращает (output_path, duration_seconds).
    """
    if len(numbers) == 0:
        raise ValueError("Нет чисел для создания аудио.")
    pcm = numbers_to_16bit_pcm(numbers)
    audio = AudioSegment(
        data=pcm,
        sample_width=SAMPLE_WIDTH,
        frame_rate=SAMPLE_RATE,
        channels=CHANNELS
    )

    # Подгонка длительности: если указали duration_sec, либо обрежем, либо дополним тишиной
    if duration_sec is not None:
        if duration_sec <= 0:
            raise ValueError("Длительность должна быть положительным числом.")
        target_ms = int(duration_sec * 1000)
        if len(audio) > target_ms:
            audio = audio[:target_ms]
        elif len(audio) < target_ms:
            silence = AudioSegment.silent(duration=(target_ms - len(audio)), frame_rate=SAMPLE_RATE)
            audio = audio + silence
        actual_duration = duration_sec
    else:
        actual_duration = len(numbers) / SAMPLE_RATE

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    output_path = os.path.join(desktop, output_name)
    audio.export(output_path, format="mp3")
    return output_path, actual_duration


# --- GUI ---
root = tk.Tk()
root.title("Конвертер массива чисел -> MP3 (с длиной)")
root.geometry("820x520")

# Инструкция
tk.Label(root, text=("Введите массив чисел (0..255) или загрузите текстовый файл со значениями, "
                       "разделёнными пробелами/запятыми. Вы также можете указать желаемую длину аудио в секундах.")).pack(pady=(12,6))

# Многострочное поле ввода
text_input = tk.Text(root, height=10, width=100)
text_input.pack(pady=6)
text_input.insert("1.0", "128 128 128 255 0 64 192")  # пример

info_label = tk.Label(root, text=("Частота дискретизации: 44100 Hz. Каждое число = 1 сэмпл. "
                                  "Длительность по умолчанию: N / 44100 сек (N = количество чисел)."))
info_label.pack(pady=4)

# Поле для ввода желаемой длины
len_frame = tk.Frame(root)
len_frame.pack(pady=6)

tk.Label(len_frame, text="Желаемая длина (секунд, опционально; если пусто — длина = N/44100):").pack(side=tk.LEFT)
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
        numbers = parse_number_array(raw_text)
    except ValueError as e:
        messagebox.showerror("Ошибка ввода", str(e))
        return
    try:
        duration = parse_duration_field()
    except ValueError as e:
        messagebox.showerror("Ошибка длины", str(e))
        return
    try:
        out_path, duration_res = create_mp3_from_numbers(numbers, duration)
    except Exception as e:
        messagebox.showerror("Ошибка при создании аудио", str(e))
        return
    messagebox.showinfo("Готово", f"MP3 сохранён:\n{out_path}\nДлительность: {duration_res:.3f} с ({len(numbers)} сэмплов)")


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
    # Поместим содержимое в поле ввода (пользователь может отредактировать)
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
        numbers = parse_number_array(data)
    except ValueError as e:
        messagebox.showerror("Ошибка парсинга", str(e))
        return
    try:
        duration = parse_duration_field()
    except ValueError as e:
        messagebox.showerror("Ошибка длины", str(e))
        return
    try:
        out_path, duration_res = create_mp3_from_numbers(numbers, duration)
    except Exception as e:
        messagebox.showerror("Ошибка при создании аудио", str(e))
        return
    # Также поместим содержимое в поле ввода
    text_input.delete('1.0', tk.END)
    text_input.insert('1.0', data)
    messagebox.showinfo("Готово", f"MP3 сохранён:\n{out_path}\nДлительность: {duration_res:.3f} с ({len(numbers)} сэмплов)")


create_btn = tk.Button(button_frame, text="Создать MP3 из поля ввода", command=on_create_from_text, width=26)
create_btn.grid(row=0, column=0, padx=8, pady=4)

load_btn = tk.Button(button_frame, text="Загрузить файл в поле ввода", command=on_load_file, width=26)
load_btn.grid(row=0, column=1, padx=8, pady=4)

load_create_btn = tk.Button(button_frame, text="Загрузить файл и создать MP3", command=on_load_file_and_create, width=26)
load_create_btn.grid(row=0, column=2, padx=8, pady=4)

quit_btn = tk.Button(root, text="Выйти", command=root.quit, width=12)
quit_btn.pack(pady=12)

root.mainloop()

