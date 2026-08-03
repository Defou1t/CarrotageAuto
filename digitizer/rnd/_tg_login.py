r"""_tg_login.py — ОДНОРАЗОВЫЙ ВХОД В TELEGRAM ПОЛЬЗОВАТЕЛЕМ (запускает ВЛАДЕЛЕЦ, не агент).

ЗАЧЕМ ИМЕННО ПОЛЬЗОВАТЕЛЬСКАЯ СЕССИЯ, А НЕ БОТ. Бот в Telegram **не может читать историю чата** —
у Bot API нет метода на выгрузку прошлого, бот видит только сообщения, пришедшие при нём. Плюс лимит
скачивания 20 МБ, а сканы здесь 20-50 МБ. Существующий `generate_telethon_session.py` логинится через
`bot_token`, то есть создаёт ботовую сессию и для этой задачи не годится.

⚠⚠ ЗАПУСКАЕТ ЭТОТ ФАЙЛ ЧЕЛОВЕК. Скрипт спросит телефон и код из Telegram — их вводите вы. Агент к
учётным данным не прикасается и в этот шаг не вмешивается; дальше он работает уже с готовым файлом
сессии, как с любым авторизованным CLI.

ЧТО НУЖНО ОДИН РАЗ:
  1. https://my.telegram.org → API development tools → создать приложение → `api_id` и `api_hash`;
  2. положить их в `E:\Carrotagki_auto\intake\tg.json`:
         {"api_id": 1234567, "api_hash": "…"}
     ⚠ Файл локальный, в чат его содержимое не отправлять и в git не класть.
  3. запустить:  python _tg_login.py
  4. ввести телефон и код (при включённой двухфакторке — ещё пароль облака).

Итог: `E:\Carrotagki_auto\intake\user.session`. Пока файл на месте, повторный вход не нужен.
"""
import sys, json, asyncio
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from telethon import TelegramClient

CFG = Path(r"E:\Carrotagki_auto\intake\tg.json")
SESSION = Path(r"E:\Carrotagki_auto\intake\user")


async def main():
    if not CFG.is_file():
        sys.exit(f"нет {CFG}\nсоздайте его: {{\"api_id\": <число>, \"api_hash\": \"<строка>\"}}\n"
                 f"api_id/api_hash берутся на https://my.telegram.org")
    c = json.loads(CFG.read_text(encoding="utf-8"))
    client = TelegramClient(str(SESSION), int(c["api_id"]), c["api_hash"])
    await client.start()                       # спросит телефон/код у ЧЕЛОВЕКА за клавиатурой
    me = await client.get_me()
    print(f"★ вход выполнен: {me.first_name or ''} @{me.username or '—'} (id {me.id})")
    print(f"★ сессия сохранена → {SESSION}.session")
    print("Больше вводить ничего не нужно: дальше работает _tg_fetch.py")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
