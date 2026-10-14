# Records the Coachwhip Crew app (crew_app.py on 127.0.0.1:8071) running one task, real time, 1280x720,
# and writes an MP4 next to this file. Run crew_app.py first. No speed-up: one video second is one real second.
import asyncio, os, glob, time, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from playwright.async_api import async_playwright
import imageio_ffmpeg

OUT = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("CREW_URL", "http://127.0.0.1:8071/")
NAME = os.environ.get("CREW_VIDEO", "coachwhip-crew-app")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1280, "height": 720}, color_scheme="dark",
                                        record_video_dir=OUT, record_video_size={"width": 1280, "height": 720})
        page = await ctx.new_page()
        await page.goto(URL)
        await page.wait_for_timeout(1500)
        t0 = time.time()
        await page.click("#go")
        await page.wait_for_function("document.querySelector('#go') && !document.querySelector('#go').disabled", timeout=1200000)
        print(f"run finished after {time.time() - t0:.0f} s")
        await page.wait_for_timeout(4000)
        await page.screenshot(path=os.path.join(OUT, NAME + "-last.png"))
        verdict = await page.evaluate("document.querySelector('#verdict').innerText")
        runner = await page.evaluate("document.querySelector('#st-run').innerText")
        print("verdict:", verdict[:120]); print("runner:", runner)
        await ctx.close(); await browser.close()
    vids = sorted(glob.glob(os.path.join(OUT, "*.webm")), key=os.path.getmtime)
    webm = vids[-1]
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = os.path.join(OUT, NAME + ".mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", mp4], check=True)
    print("mp4:", mp4, os.path.getsize(mp4) // 1024, "KB")

asyncio.run(main())
