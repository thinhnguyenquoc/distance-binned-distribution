// ==UserScript==
// @name         BCH Autoplay Bot
// @namespace    http://tampermonkey.net/
// @version      1.8
// @description  Tự động quản lý cược BCH theo batch 10 lượt autoplay và kiểm soát trần cược an toàn
// @match        *://*/*
// @grant        none
// ==/UserScript==

(function () {
    'use strict';

    let isProcessing = false;
    let upBal = 0;
    let lastBatchBal = 0;
    let hasRunAtLeastOnce = false;
    const MIN_BET = 1e-7; // 10e-8 = 1e-7 BCH

    const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const round8 = (num) => Math.round(num * 1e8) / 1e8;

    async function clickRepeatedly(element, times, interval = 300) {
        if (!element || times <= 0) return;
        for (let i = 0; i < times; i++) {
            element.click();
            if (i < times - 1) await delay(interval);
        }
    }

    function findButtonByText(text) {
        const buttons = document.querySelectorAll('button');
        for (const btn of buttons) {
            if (btn.textContent.trim() === text) return btn;
        }
        return null;
    }

    function findPlayButton() {
        const buttons = document.querySelectorAll('button');
        for (const btn of buttons) {
            const text = btn.textContent.trim();
            if (text.startsWith('PLAY') && text.includes('ON MAX WIN')) return btn;
        }
        return null;
    }

    function isAutoplayRunning() {
        const buttons = document.querySelectorAll('button');
        for (const btn of buttons) {
            if (btn.textContent.trim() === 'CANCEL') return true;
        }
        return false;
    }

    // Tự động nhận Giveaway độc lập, không phụ thuộc vào trạng thái nút Play hay autoplay
    function checkJoinGiveaway() {
        try {
            const joinBtn = document.querySelector('[aria-label="Join giveaway"]');
            if (!joinBtn) return;

            const hasCountdown = Array.from(document.querySelectorAll('div'))
                .some(el => el.textContent.includes('Next giveaway'));

            if (!hasCountdown) {
                console.log('[BCH-BOT] Phát hiện Giveaway khả dụng -> Click Join Giveaway!');
                joinBtn.click();
            }
        } catch (e) {
            console.error('[BCH-BOT] Lỗi khi check giveaway:', e);
        }
    }

    // Kiểm tra định kỳ mỗi 5 giây phòng trường hợp DOM không thay đổi
    setInterval(checkJoinGiveaway, 5000);

    const observer = new MutationObserver(async () => {
        // Kiểm tra giveaway ngay lập tức khi DOM có bất kỳ thay đổi nào
        checkJoinGiveaway();

        if (isProcessing) return;

        // Nếu đang trong tiến trình autoplay (nút CANCEL đang có), không can thiệp
        if (isAutoplayRunning()) return;

        const playBtn = findPlayButton();
        if (!playBtn) return;

        // 1. Kiểm tra panel big wins
        const isBigWins = Array.from(document.querySelectorAll('aside div, div'))
            .some(el => el.children.length === 0 && el.textContent.includes('big wins'));
        if (!isBigWins) return;

        // 2. Đọc số dư BCH từ header
        const bchHeader = Array.from(document.querySelectorAll('header'))
            .find(el => el.textContent.includes('BCH'));
        if (!bchHeader) return;

        const match = bchHeader.textContent.replace(/,/g, '').match(/[\d.]+/);
        if (!match) return;

        const bal = round8(parseFloat(match[0]));
        const betElement = document.querySelector('[aria-label="stake"]');
        const increase = document.querySelector('[aria-label="increase stake"]');
        const decrease = document.querySelector('[aria-label="decrease stake"]');
        const openPopupBtn = playBtn.querySelector('svg')?.parentElement;

        if (!betElement || !openPopupBtn) return;

        let bet = round8(parseFloat(betElement.value));
        if (Number.isNaN(bal) || Number.isNaN(bet)) return;

        isProcessing = true;

        try {
            console.log(`[BCH-BOT] Bal: ${bal} | UpBal: ${upBal} | LastBatchBal: ${lastBatchBal} | Bet: ${bet}`);

            if (!hasRunAtLeastOnce) {
                upBal = bal;
                lastBatchBal = bal;
            } else {
                const batchProfit = round8(bal - lastBatchBal);
                console.log(`[BCH-BOT] 10-Bet Batch kết thúc! Profit: ${batchProfit}`);

                if (bal > upBal) {
                    // Đạt đỉnh vốn mới: Cập nhật đỉnh và reset về MIN_BET
                    console.log(`[BCH-BOT] Đạt đỉnh mới: ${bal} > ${upBal}. Giảm về MIN_BET.`);
                    upBal = bal;

                    let tempBet = bet;
                    let decreaseCount = 0;
                    while (round8(tempBet / 2) >= MIN_BET) {
                        decreaseCount++;
                        tempBet = round8(tempBet / 2);
                    }

                    if (decreaseCount > 0) {
                        await clickRepeatedly(decrease, decreaseCount, 300);
                        bet = tempBet;
                    }
                } else if (batchProfit < 0) {
                    // Kiểm tra trước xem x2 có an toàn không
                    const nextBet = round8(bet * 2);
                    if (round8(2 * nextBet + bal) <= upBal) {
                        console.log(`[BCH-BOT] Batch lỗ (${batchProfit} BCH) -> x2 Bet lên ${nextBet}.`);
                        await delay(300);
                        increase?.click();
                        bet = nextBet;
                    } else {
                        console.log(`[BCH-BOT] Batch lỗ nhưng x2 sẽ vượt trần an toàn (${round8(2 * nextBet + bal)} > ${upBal}) -> Không x2.`);
                        
                        // Nếu bản thân mức cược hiện tại đã vượt trần: 2*bet + bal > upBal thì hạ cược
                        if (round8(2 * bet + bal) > upBal && bet > MIN_BET) {
                            let tempBet = bet;
                            let decreaseCount = 0;
                            while (round8(2 * tempBet + bal) > upBal && round8(tempBet / 2) >= MIN_BET) {
                                decreaseCount++;
                                tempBet = round8(tempBet / 2);
                            }
                            if (decreaseCount > 0) {
                                console.log(`[BCH-BOT] Hạ cược ${decreaseCount} lần về ${tempBet} để nằm trong trần an toàn.`);
                                await delay(300);
                                await clickRepeatedly(decrease, decreaseCount, 300);
                                bet = tempBet;
                            }
                        }
                    }
                } else {
                    // Thắng / hòa: Giữ nguyên cược, chỉ hạ nếu mức cược hiện tại vượt trần an toàn
                    console.log(`[BCH-BOT] Batch hòa/thắng (+${batchProfit} BCH) -> Giữ nguyên bet.`);
                    if (round8(2 * bet + bal) > upBal && bet > MIN_BET) {
                        let tempBet = bet;
                        let decreaseCount = 0;
                        while (round8(2 * tempBet + bal) > upBal && round8(tempBet / 2) >= MIN_BET) {
                            decreaseCount++;
                            tempBet = round8(tempBet / 2);
                        }
                        if (decreaseCount > 0) {
                            console.log(`[BCH-BOT] Hạ cược an toàn ${decreaseCount} lần về ${tempBet}.`);
                            await delay(300);
                            await clickRepeatedly(decrease, decreaseCount, 300);
                            bet = tempBet;
                        }
                    }
                }
            }

            // Nghỉ khoảng ~1.5s sau khi phát hiện nút Play sáng trước khi mở popup cho tự nhiên giống người
            const openPopupDelay = 1400 + Math.floor(Math.random() * 300); // 1.4s - 1.7s (~1.5s)
            console.log(`[BCH-BOT] Chờ ${openPopupDelay}ms trước khi bấm mở popup autoplay...`);
            await delay(openPopupDelay);

            // Mở popup cấu hình Autoplay
            openPopupBtn.click();

            // Chờ popup mở và kiểm tra input #number-of-games__0 có giá trị là '10'
            let gamesInputValid = false;
            for (let i = 0; i < 16; i++) {
                await delay(300);
                const gamesInput = document.getElementById('number-of-games__0');
                if (gamesInput && gamesInput.value.trim() === '10') {
                    gamesInputValid = true;
                    break;
                }
            }

            if (!gamesInputValid) {
                console.warn('[BCH-BOT] Input number-of-games__0 chưa phải là 10, đóng/bỏ qua popup để an toàn.');
                // Đóng popup nếu cần (click lại openPopupBtn hoặc nút đóng nếu có)
                return;
            }

            // Đợi nút START AUTOPLAY xuất hiện
            let startBtn = null;
            for (let i = 0; i < 16; i++) {
                startBtn = findButtonByText('START AUTOPLAY');
                if (startBtn) break;
                await delay(300);
            }

            if (startBtn) {
                // Chờ khoảng ~2s sau khi mở popup và cấu hình xong trước khi bấm Start cho tự nhiên giống người thao tác
                const humanDelay = 1800 + Math.floor(Math.random() * 400); // 1.8s - 2.2s
                console.log(`[BCH-BOT] Chờ ${humanDelay}ms trước khi bấm START AUTOPLAY (human-like delay)...`);
                await delay(humanDelay);

                // Cập nhật lại số dư mới nhất (nếu có biến động nhẹ trong 2s chờ) trước khi bấm Start
                const latestBchHeader = Array.from(document.querySelectorAll('header'))
                    .find(el => el.textContent.includes('BCH'));
                if (latestBchHeader) {
                    const latestMatch = latestBchHeader.textContent.replace(/,/g, '').match(/[\d.]+/);
                    if (latestMatch) lastBatchBal = round8(parseFloat(latestMatch[0]));
                } else {
                    lastBatchBal = bal;
                }

                hasRunAtLeastOnce = true;
                startBtn.click();
                console.log(`[BCH-BOT] Đã bấm START AUTOPLAY 10 lượt với bet = ${bet}`);

                // Bước 1: Chờ autoplay thực sự bắt đầu (nút CANCEL xuất hiện) - tối đa 6s
                let started = false;
                for (let i = 0; i < 20; i++) {
                    await delay(300);
                    if (isAutoplayRunning()) {
                        started = true;
                        break;
                    }
                }

                if (started) {
                    console.log('[BCH-BOT] Autoplay đang chạy... Chờ hoàn tất 10 lượt.');
                    // Bước 2: Chờ cho tới khi CANCEL biến mất và nút PLAY xuất hiện lại
                    while (isAutoplayRunning() || !findPlayButton()) {
                        await delay(500);
                    }
                    console.log('[BCH-BOT] 10 lượt đã hoàn tất!');
                    // Nghỉ 1s để giao diện cập nhật chính xác số dư mới
                    await delay(1000);
                } else {
                    console.warn('[BCH-BOT] Autoplay không vào trạng thái CANCEL kịp thời, chờ thêm 2s trước khi kết thúc chu kỳ.');
                    await delay(2000);
                }
            }
        } catch (err) {
            console.error('[BCH-BOT] Lỗi khi thực thi:', err);
        } finally {
            isProcessing = false;
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
})();
