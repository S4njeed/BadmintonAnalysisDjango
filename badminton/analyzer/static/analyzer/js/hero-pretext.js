const HERO_SELECTOR = '[data-pretext-hero]';
const TITLE_SELECTOR = '[data-pretext-hero-title]';

let resizeRaf = null;
let resizeObserver = null;

function getNumericFontValue(value, fallback) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function clearHeroWidth(heroElement) {
    heroElement.style.removeProperty('--hero-title-width');
}

async function balanceHeroTitle() {
    const heroElement = document.querySelector(HERO_SELECTOR);
    const titleElement = document.querySelector(TITLE_SELECTOR);

    if (!heroElement || !titleElement) {
        return;
    }

    if (window.innerWidth < 768) {
        clearHeroWidth(heroElement);
        return;
    }

    try {
        const module = await import('https://esm.sh/@chenglou/pretext?bundle');
        const prepareWithSegments = module.prepareWithSegments;
        const layoutWithLines = module.layoutWithLines;

        if (document.fonts && document.fonts.ready) {
            await document.fonts.ready;
        }

        const computedStyle = window.getComputedStyle(titleElement);
        const fontSize = getNumericFontValue(computedStyle.fontSize, 16);
        const lineHeight = getNumericFontValue(computedStyle.lineHeight, fontSize * 1.05);
        const font = `${computedStyle.fontWeight} ${computedStyle.fontSize} ${computedStyle.fontFamily}`;
        const text = (titleElement.textContent || '').trim();

        if (!text) {
            clearHeroWidth(heroElement);
            return;
        }

        const prepared = prepareWithSegments(text, font);
        const availableWidth = Math.min(
            Math.max(heroElement.clientWidth - 8, 360),
            920,
        );

        let low = 360;
        let high = availableWidth;
        let bestWidth = availableWidth;

        while (low <= high) {
            const candidate = Math.round((low + high) / 2);
            const lineCount = layoutWithLines(prepared, candidate, lineHeight).lineCount;

            if (lineCount <= 2) {
                bestWidth = candidate;
                high = candidate - 1;
            } else {
                low = candidate + 1;
            }
        }

        heroElement.style.setProperty('--hero-title-width', `${bestWidth}px`);
    } catch (error) {
        console.debug('Pretext hero balancing skipped', error);
        clearHeroWidth(heroElement);
    }
}

function scheduleBalance() {
    if (resizeRaf !== null) {
        window.cancelAnimationFrame(resizeRaf);
    }

    resizeRaf = window.requestAnimationFrame(() => {
        resizeRaf = null;
        balanceHeroTitle();
    });
}

function observeHero() {
    const heroElement = document.querySelector(HERO_SELECTOR);

    if (!heroElement || !('ResizeObserver' in window)) {
        return;
    }

    resizeObserver = new ResizeObserver(() => {
        scheduleBalance();
    });
    resizeObserver.observe(heroElement);
}

window.addEventListener('DOMContentLoaded', () => {
    scheduleBalance();
    observeHero();
});

window.addEventListener('resize', scheduleBalance, { passive: true });
window.addEventListener('orientationchange', scheduleBalance, { passive: true });
