/**
 * 对比页逻辑：加载随机配对、提交偏好、更新状态。
 */

(function () {
  "use strict";

  const els = {
    currentRound: document.getElementById("currentRound"),
    maxRound: document.getElementById("maxRound"),
    remaining: document.getElementById("remaining"),
    totalFaces: document.getElementById("totalFaces"),
    leftImage: document.getElementById("leftImage"),
    rightImage: document.getElementById("rightImage"),
    leftName: document.getElementById("leftName"),
    rightName: document.getElementById("rightName"),
    btnLikeLeft: document.getElementById("btnLikeLeft"),
    btnLikeRight: document.getElementById("btnLikeRight"),
    comparePanel: document.getElementById("comparePanel"),
    messageBox: document.getElementById("messageBox"),
  };

  /** 当前展示的图片 ID */
  let currentPair = { left_id: null, right_id: null };
  let voting = false;

  function showMessage(text, type) {
    els.messageBox.hidden = false;
    els.messageBox.textContent = text;
    els.messageBox.className = "message-box " + (type || "info");
  }

  function hideMessage() {
    els.messageBox.hidden = true;
    els.messageBox.textContent = "";
    els.messageBox.className = "message-box";
  }

  function updateStatus(status) {
    if (!status) return;
    els.currentRound.textContent = status.current_round;
    if (status.unlimited_iterations) {
      els.maxRound.textContent = "∞";
      els.remaining.textContent = "∞";
    } else {
      els.maxRound.textContent = status.max_iterations;
      els.remaining.textContent = status.remaining;
    }
    els.totalFaces.textContent = status.total_faces;
  }

  function setButtonsEnabled(enabled) {
    els.btnLikeLeft.disabled = !enabled;
    els.btnLikeRight.disabled = !enabled;
    els.comparePanel.classList.toggle("disabled", !enabled);
  }

  function bindImageError(img) {
    img.onerror = function () {
      img.classList.add("error");
      img.alt = "图片加载失败";
    };
    img.onload = function () {
      img.classList.remove("error");
    };
  }

  bindImageError(els.leftImage);
  bindImageError(els.rightImage);

  async function fetchPair() {
    setButtonsEnabled(false);
    hideMessage();

    try {
      const resp = await fetch("/api/pair");
      const data = await resp.json();
      updateStatus(data.status);

      if (!data.ok) {
        showMessage(data.error || "无法加载对比组", "warning");
        return;
      }

      currentPair = {
        left_id: data.left.id,
        right_id: data.right.id,
      };

      els.leftImage.src = data.left.url + "?t=" + Date.now();
      els.rightImage.src = data.right.url + "?t=" + Date.now();
      els.leftName.textContent = data.left.id;
      els.rightName.textContent = data.right.id;

      setButtonsEnabled(true);
    } catch (err) {
      showMessage("网络错误: " + err.message, "error");
    }
  }

  async function submitVote(choice) {
    if (voting || !currentPair.left_id) return;
    voting = true;
    setButtonsEnabled(false);

    try {
      const resp = await fetch("/api/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_id: currentPair.left_id,
          right_id: currentPair.right_id,
          choice: choice,
        }),
      });

      const data = await resp.json();
      updateStatus(data.status);

      if (!data.ok) {
        showMessage(data.error || "提交失败", "error");
        setButtonsEnabled(true);
        voting = false;
        return;
      }

      await fetchPair();
    } catch (err) {
      showMessage("提交失败: " + err.message, "error");
      setButtonsEnabled(true);
    } finally {
      voting = false;
    }
  }

  els.btnLikeLeft.addEventListener("click", function () {
    submitVote("left");
  });

  els.btnLikeRight.addEventListener("click", function () {
    submitVote("right");
  });

  document.addEventListener("keydown", function (e) {
    if (voting || els.btnLikeLeft.disabled) return;
    if (e.key === "ArrowLeft") submitVote("left");
    if (e.key === "ArrowRight") submitVote("right");
  });

  fetchPair();
})();
