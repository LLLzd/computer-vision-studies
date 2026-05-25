/**
 * 结果页逻辑：展示排名、导出 CSV、重置数据。
 */

(function () {
  "use strict";

  const els = {
    totalComparisons: document.getElementById("totalComparisons"),
    totalFaces: document.getElementById("totalFaces"),
    sessionStatus: document.getElementById("sessionStatus"),
    resultsBody: document.getElementById("resultsBody"),
    messageBox: document.getElementById("messageBox"),
    btnExport: document.getElementById("btnExport"),
    btnReset: document.getElementById("btnReset"),
  };

  function showMessage(text, type) {
    els.messageBox.hidden = false;
    els.messageBox.textContent = text;
    els.messageBox.className = "message-box " + (type || "info");
  }

  function rankClass(rank) {
    if (rank === 1) return "rank-gold";
    if (rank === 2) return "rank-silver";
    if (rank === 3) return "rank-bronze";
    return "";
  }

  function renderTable(rankings) {
    if (!rankings || rankings.length === 0) {
      els.resultsBody.innerHTML =
        '<tr><td colspan="8" class="empty-row">暂无数据，请先将图片放入 static/faces 并完成对比</td></tr>';
      return;
    }

    els.resultsBody.innerHTML = rankings
      .map(function (item) {
        return (
          "<tr>" +
          '<td class="' +
          rankClass(item.rank) +
          '">#' +
          item.rank +
          "</td>" +
          '<td><img class="thumb" src="' +
          item.url +
          '" alt="' +
          item.id +
          '" onerror="this.classList.add(\'error\')"></td>' +
          "<td>" +
          item.id +
          "</td>" +
          '<td><span class="score-badge">' +
          item.score.toFixed(1) +
          "</span></td>" +
          "<td>" +
          item.effective_score.toFixed(4) +
          "</td>" +
          "<td>" +
          item.mu.toFixed(4) +
          "</td>" +
          "<td>" +
          item.sigma.toFixed(4) +
          "</td>" +
          "<td>" +
          item.comparisons +
          "</td>" +
          "</tr>"
        );
      })
      .join("");
  }

  async function loadRankings() {
    try {
      const resp = await fetch("/api/rankings");
      const data = await resp.json();

      if (!data.ok) {
        showMessage("加载失败", "error");
        return;
      }

      els.totalComparisons.textContent = data.total_comparisons;
      els.totalFaces.textContent = data.status.total_faces;

      const status = data.status;
      if (status.current_round > 0) {
        if (status.unlimited_iterations) {
          els.sessionStatus.textContent = "进行中 (" + status.current_round + "/∞)";
        } else {
          els.sessionStatus.textContent =
            "进行中 (" + status.current_round + "/" + status.max_iterations + ")";
        }
        if (status.converged) {
          els.sessionStatus.textContent += " · 已收敛";
        }
      } else {
        els.sessionStatus.textContent = "未开始";
      }

      renderTable(data.rankings);
    } catch (err) {
      showMessage("加载失败: " + err.message, "error");
    }
  }

  function exportCsv() {
    window.location.href = "/api/export/csv";
  }

  async function resetAll() {
    if (
      !confirm(
        "确定要重置所有排名与对比记录吗？此操作不可撤销（图片文件不会被删除）。"
      )
    ) {
      return;
    }

    els.btnReset.disabled = true;

    try {
      const resp = await fetch("/api/reset", { method: "POST" });
      const data = await resp.json();

      if (data.ok) {
        showMessage("数据已重置", "success");
        await loadRankings();
      } else {
        showMessage("重置失败", "error");
      }
    } catch (err) {
      showMessage("重置失败: " + err.message, "error");
    } finally {
      els.btnReset.disabled = false;
    }
  }

  els.btnExport.addEventListener("click", exportCsv);
  els.btnReset.addEventListener("click", resetAll);

  loadRankings();
})();
