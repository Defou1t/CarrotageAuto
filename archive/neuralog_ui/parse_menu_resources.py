"""Скрипт глубокого анализа секций NDSLOG.EXE и поиска внешних ресурсов."""
import sys
from pathlib import Path

try:
    import pefile
except ImportError:
    print("pip install pefile")
    sys.exit(1)

EXE = Path(r"F:\nds\bin\NDSLOG.EXE")

def main():
    print("=== ГЛУБОКИЙ АНАЛИЗ СТРУКТУРЫ И СЕКЦИЙ ФАЙЛА ===\n")
    if not EXE.exists():
        print(f"❌ Файл не найден по пути: {EXE}")
        return
        
    # 1. Проверяем физический размер файла на диске
    file_size = EXE.stat().st_size
    print(f"Физический размер файла на диске: {file_size} байт ({file_size / (1024*1024):.2f} МБ)")
    
    # 2. Анализируем PE-секции на предмет упаковщиков (UPX и т.д.)
    try:
        pe = pefile.PE(str(EXE))
        print("\nСекции исполняемого файла:")
        for section in pe.sections:
            name = section.Name.decode('ansi', errors='ignore').strip('\x00')
            vaddr = hex(section.VirtualAddress)
            vsize = section.Misc_VirtualSize
            raw_size = section.SizeOfRawData
            print(f"  Секция: {name:8} | RVA: {vaddr:10} | Вирт. размер: {vsize:8} | На диске: {raw_size:8}")
    except Exception as e:
        print(f"❌ Ошибка анализа PE: {e}")

    # 3. Ищем соседние DLL, которые могут содержать реальные меню (Локализация/Ресурсы)
    print("\nПроверяем наличие библиотек ресурсов в папке приложения:")
    bin_dir = EXE.parent
    found_dlls = False
    for f in bin_dir.glob("*"):
        if f.suffix.lower() in ['.dll', '.res'] or 'res' in f.name.lower():
            print(f"  📦 Найден файл рядом: {f.name} ({f.stat().st_size / 1024:.1f} КБ)")
            found_dlls = True
    if not found_dlls:
        print("  Рядом с EXE-файлом сопутствующих DLL или файлов ресурсов не обнаружено.")
            
    # 4. Сканируем ВЕСЬ файл целиком на наличие строк "LAS" или "Export"
    print("\nСканируем весь бинарник на упоминание ключевых слов...")
    with open(EXE, "rb") as f:
        content = f.read()
        
    keywords = [b"Export", b"LAS", b"Save As", b"Log"]
    print("  [Поиск в кодировке ASCII/ANSI]:")
    for kw in keywords:
        count = content.count(kw)
        print(f"    Слово '{kw.decode()}' встретилось: {count} раз(а)")
        
    print("  [Поиск в кодировке UTF-16 (Unicode)]:")
    for kw in keywords:
        kw_u = kw.decode().encode('utf-16-le')
        count = content.count(kw_u)
        print(f"    Слово '{kw.decode()}' встретилось: {count} раз(а)")

if __name__ == "__main__":
    main()