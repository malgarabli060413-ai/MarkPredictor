
// === SCORE SUMMARY PAGE ===
const params = new URLSearchParams(window.location.search);
const score = parseFloat(params.get("score")) || 0;

const scoreNumber = document.getElementById("scoreNumber");
const summaryText = document.querySelector(".summary-text p:last-of-type");

// If scoreNumber element exists, update it
if (scoreNumber) {
  scoreNumber.textContent = score.toFixed(1);

  // Style based on performance
  if (score >= 75) {
    scoreNumber.style.color = "green";
  } else if (score >= 50) {
    scoreNumber.style.color = "#f59e0b";
  } else {
    scoreNumber.style.color = "red";
  }

  // Personalized feedback message
  if (score < 50) {
    summaryText.textContent =
      "📉 Your score suggests there’s room to improve your consistency and study habits.";
  } else if (score < 75) {
    summaryText.textContent =
      "👍 Good effort! You're on the right track — focus on regular revision and balance.";
  } else {
    summaryText.textContent =
      "🚀 Excellent work! You’ve demonstrated strong academic potential — keep it up!";
  }
}
