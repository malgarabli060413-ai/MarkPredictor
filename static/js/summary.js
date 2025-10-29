
// === SCORE SUMMARY PAGE ===
const params = new URLSearchParams(window.location.search);
const score = parseFloat(params.get("score")) || parseFloat(sessionStorage.getItem('predictedScore')) || 0;
const performanceLevel = sessionStorage.getItem('performanceLevel') || '';

const scoreNumber = document.getElementById("scoreNumber");
const summaryText = document.querySelector(".summary-text p:last-of-type");

// If scoreNumber element exists, update it
if (scoreNumber) {
  scoreNumber.textContent = Math.round(score);

  // Style based on performance
  if (score >= 80) {
    scoreNumber.style.color = "#22c55e"; // green
  } else if (score >= 70) {
    scoreNumber.style.color = "#3b82f6"; // blue
  } else if (score >= 60) {
    scoreNumber.style.color = "#f59e0b"; // orange
  } else {
    scoreNumber.style.color = "#ef4444"; // red
  }

  // Personalized feedback message based on ML prediction
  if (performanceLevel === "Excellent") {
    summaryText.textContent =
      "🌟 Outstanding! Your predicted score shows exceptional academic performance. Keep up your excellent study habits!";
  } else if (performanceLevel === "Very Good") {
    summaryText.textContent =
      "🎯 Great job! You've shown strong understanding in key areas. Let's explore how to fine-tune your performance further.";
  } else if (performanceLevel === "Good") {
    summaryText.textContent =
      "👍 Good effort! You're on the right track. Focus on consistent study habits and maintaining balance.";
  } else if (performanceLevel === "Average") {
    summaryText.textContent =
      "📚 You're doing okay! There's room for improvement. Consider increasing study hours and improving attendance.";
  } else {
    summaryText.textContent =
      "💪 Don't give up! Your score suggests areas for improvement. Let's work on building better study habits together.";
  }
}
