from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import requests, os
import psycopg2
from psycopg2.extras import RealDictCursor
from config import ROOMS, BOT_TOKEN, AI_URL, ALLOWED_CHAT_IDS

if BOT_TOKEN is None:
    raise SystemExit("BOT_TOKEN environment variable not set")

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise SystemExit("DATABASE_URL environment variable not set")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

def is_authorized(user_id: int) -> bool:
    """Check if user is authorized to use the bot"""
    if not ALLOWED_CHAT_IDS:
        return True  # If no restrictions configured, allow all
    return user_id in ALLOWED_CHAT_IDS

# Initialize database
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS radiators(
        room TEXT PRIMARY KEY, 
        level REAL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()
cur.close()

def get_db():
    """Get a new database connection"""
    return psycopg2.connect(DATABASE_URL)

@dp.message_handler(commands=['start'])
async def start(msg: types.Message):
    """Welcome message with menu button"""
    if not is_authorized(msg.from_user.id):
        await msg.reply("⛔ Du har inte behörighet att använda denna bot.")
        return
    
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(types.KeyboardButton("🔧 Ställ in element"))
    keyboard.add(types.KeyboardButton("📊 Status"))
    
    await msg.reply(
        "🏠 *Smart Element Assistent*\n\n"
        "Styr dina element med AI-driven temperaturhantering.\n\n"
        "Använd menyn nedan eller skriv /set",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )

@dp.message_handler(lambda msg: msg.text == "🔧 Ställ in element")
@dp.message_handler(commands=['set'])
async def set_radiator(msg: types.Message):
    if not is_authorized(msg.from_user.id):
        await msg.reply("⛔ Du har inte behörighet att använda denna bot.")
        return
    
    kb = types.InlineKeyboardMarkup(row_width=2)
    buttons = [
        types.InlineKeyboardButton(text=room, callback_data=f"room:{room}")
        for room in ROOMS
    ]
    kb.add(*buttons)
    await msg.reply("*Välj rum:*", reply_markup=kb, parse_mode="Markdown")

@dp.message_handler(lambda msg: msg.text == "📊 Status")
async def status(msg: types.Message):
    """Show current radiator levels"""
    if not is_authorized(msg.from_user.id):
        await msg.reply("⛔ Du har inte behörighet att använda denna bot.")
        return
    
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT room, level, updated_at FROM radiators ORDER BY room")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows:
            await msg.reply("Inga elementnivåer inställda än. Använd 🔧 Ställ in element för att konfigurera.")
            return
        
        status_text = "📊 *Aktuella elementnivåer:*\n\n"
        for room, level, updated in rows:
            target = ROOMS.get(room, {}).get("target", "?")
            status_text += f"🌡️ *{room}*: Nivå {level} (Måltemp {target}°C)\n"
            status_text += f"   _Uppdaterad: {updated.strftime('%H:%M %d/%m')}_\n\n"
        
        await msg.reply(status_text, parse_mode="Markdown")
    except Exception as e:
        await msg.reply(f"⚠️ Fel vid hämtning av status: {str(e)}")

@dp.callback_query_handler(lambda c: c.data and c.data.startswith("room:"))
async def choose_level(callback: types.CallbackQuery):
    if not is_authorized(callback.from_user.id):
        await callback.answer("⛔ Du har inte behörighet att använda denna bot.", show_alert=True)
        return
    
    room = callback.data.split(":", 1)[1]
    scale = ROOMS[room]["scale"]
    target = ROOMS[room]["target"]
    
    # Get current level from database
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT level FROM radiators WHERE room = %s", (room,))
        result = cur.fetchone()
        current = result[0] if result else 0
        cur.close()
        conn.close()
    except:
        current = 0
    
    # Create keyboard with 3 columns
    kb = types.InlineKeyboardMarkup(row_width=3)
    buttons = []
    for lvl in scale:
        # Highlight current level
        text = f"✓ {lvl}" if lvl == current else str(lvl)
        buttons.append(types.InlineKeyboardButton(text, callback_data=f"set:{room}:{lvl}"))
    kb.add(*buttons)
    
    await callback.message.edit_text(
        f"*{room}*\n"
        f"Måltemp: {target}°C\n"
        f"Aktuell nivå: {current}\n\n"
        f"Välj nivå:",
        reply_markup=kb,
        parse_mode="Markdown"
    )
    await callback.answer()

@dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:"))
async def confirm(callback: types.CallbackQuery):
    if not is_authorized(callback.from_user.id):
        await callback.answer("⛔ Du har inte behörighet att använda denna bot.", show_alert=True)
        return
    
    _, room, lvl = callback.data.split(":")
    lvl = float(lvl)
    
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO radiators (room, level) VALUES (%s, %s) ON CONFLICT (room) DO UPDATE SET level = %s, updated_at = CURRENT_TIMESTAMP",
            (room, lvl, lvl)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Saved {room} = {lvl} to database")
    except Exception as e:
        print(f"❌ Database error: {e}")
        await callback.message.edit_text(f"⚠️ Database error: {str(e)}")
        await callback.answer()
        return
    
    await callback.message.edit_text(f"✅ {room} element inställt på {lvl} (måltemp {ROOMS[room]['target']}°C)")
    await callback.answer()

if __name__ == "__main__":
    executor.start_polling(dp)
