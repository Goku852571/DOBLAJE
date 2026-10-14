"""Adapted internal TikTok module; console CLI and external paths removed."""
from .runtime import paths
from core.media import run_cmd
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, BrowserContext, Page
from playwright_stealth import Stealth

CHROME_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--start-maximized",
    "--no-sandbox",
    "--disable-infobars",
    "--disable-dev-shm-usage",
    "--lang=es-ES,es",
]

def obtener_contexto_persistente(headless: bool = False) -> tuple:
    """
    Inicia un contexto persistente de Playwright utilizando el ejecutable real de Google Chrome.
    Esto minimiza la detección anti-bot y guarda cookies y sesión.
    """
    profile = paths().profile
    profile.mkdir(parents=True, exist_ok=True)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(paths().root / "data" / "browsers")
    p = sync_playwright().start()

    try:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile),
            channel="chrome" if paths().browser == "chrome" else None,
            headless=headless,
            args=CHROME_ARGS,
            no_viewport=True,
            ignore_default_args=["--enable-automation"],
        )
    
        page = context.pages[0] if context.pages else context.new_page()
        Stealth().apply_stealth_sync(page)
    
    except Exception:
        p.stop()
        raise
    return p, context, page

def manejar_validaciones_y_modales(page: Page, timeout_ms: int = 4000) -> dict:
    """
    Inspecciona la página en busca de modales invasivos conocidos de TikTok:
    - Banner de Cookies
    - Modal de Inicio de Sesión
    - Modal de Verificación Captcha / Rompecabezas
    - Modal de Notificación de App
    """
    resultados = {
        "cookies_detectado": False,
        "login_modal_detectado": False,
        "captcha_detectado": False,
        "accion_realizada": []
    }

    page.wait_for_timeout(1000)

    # 1. Detección de Captcha / Verificación humana
    captcha_selectors = [
        "#captcha-verify-image",
        ".captcha_verify_container",
        "div[class*='captcha']",
        "div[class*='verify-center']"
    ]
    for sel in captcha_selectors:
        try:
            if page.locator(sel).count() > 0 and page.locator(sel).first.is_visible(timeout=500):
                resultados["captcha_detectado"] = True
                resultados["accion_realizada"].append("¡ALERTA! Se detectó captcha de validación de TikTok.")
                break
        except Exception:
            pass

    # 2. Detección de Banner de Cookies
    cookie_buttons = [
        "button:has-text('Rechazar todo')",
        "button:has-text('Rechazar todas')",
        "button:has-text('Aceptar todas')",
        "button:has-text('Aceptar todo')",
        "[data-e2e='cookie-banner'] button",
    ]
    for sel in cookie_buttons:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=800):
                resultados["cookies_detectado"] = True
                btn.click()
                resultados["accion_realizada"].append(f"Banner de cookies cerrado con selector: {sel}")
                page.wait_for_timeout(500)
                break
        except Exception:
            pass

    # 3. Detección y cierre de Modal de Inicio de Sesión
    login_close_selectors = [
        "div[data-e2e='modal-close-icon']",
        "[aria-label='Cerrar']",
        "[aria-label='Close']",
        "button[data-e2e='close-icon']",
        "div[role='dialog'] svg",
    ]
    for sel in login_close_selectors:
        try:
            close_btn = page.locator(sel).first
            if close_btn.is_visible(timeout=800):
                resultados["login_modal_detectado"] = True
                close_btn.click()
                resultados["accion_realizada"].append(f"Modal de login cerrado con selector: {sel}")
                page.wait_for_timeout(500)
                break
        except Exception:
            pass

    # Si aún hay modal abierto bloqueando, intentar cerrar con tecla Escape
    try:
        dialog = page.locator("div[role='dialog'], [data-e2e='modal']").first
        if dialog.is_visible(timeout=500):
            page.keyboard.press("Escape")
            resultados["accion_realizada"].append("Se presionó tecla 'Escape' para cerrar modal persistente.")
            page.wait_for_timeout(500)
    except Exception:
        pass

    return resultados




def iniciar_sesion():
    """Save the internal persistent session when the user closes the browser."""
    from playwright.sync_api import Error
    p, context, page = obtener_contexto_persistente(headless=False)
    try:
        page.goto("https://www.tiktok.com/login", timeout=60000)
        print("Inicia sesión en el navegador y cierra sus ventanas para guardar la sesión.", flush=True)
        while context.pages:
            try:
                context.pages[0].wait_for_timeout(500)
            except Error:
                if not context.pages:
                    break
                raise
    finally:
        try:
            context.close()
        finally:
            p.stop()
