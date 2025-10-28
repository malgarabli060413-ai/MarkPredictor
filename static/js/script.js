// Dynamic year in footer
document.getElementById("year").textContent = new Date().getFullYear();

// Calculate dummy score
//document.getElementById("calcBtn").addEventListener("click", () => {
//    const randomScore = Math.floor(60 + Math.random() * 40);
//    document.getElementById("scoreNumber").textContent = randomScore;
//    alert(`Your predicted score is ${randomScore}/100`);
//});

//document.getElementsByClassName("form-grid").addEventListener("click", () => {
//    const gender =
//})

// === GRADE POTENTIAL CALCULATOR ===
function calculateGradePotential(data) {
  let studyHours = parseFloat(data.studyHours) || 0;
  let attendance = parseFloat(data.attendance) || 0;
  let sleepHours = parseFloat(data.sleepHours) || 0;
  let socialHours = parseFloat(data.socialMediaHours) || 0;
  let mentalState = data.mentalState;
  let parentEdu = data.parentEdu;

  // Normalize inputs
  studyHours = Math.min(studyHours, 50);
  attendance = Math.min(attendance, 100);
  sleepHours = Math.min(sleepHours, 10);
  socialHours = Math.min(socialHours, 10);

  const mentalFactor = {
    "Stupid": 0.3,
    "Normal": 0.6,
    "Einstein": 1.0
  }[mentalState] || 0.5;

  const parentFactor = {
    "High-School Diploma": 0.5,
    "Bachelor's Degree": 0.7,
    "Doctorate": 0.9
  }[parentEdu] || 0.6;

  const wellness = ((sleepHours / 8) - (socialHours / 10)) * 100;

  const potential =
    (studyHours / 50) * 35 + // study = 35%
    (attendance / 100) * 25 + // attendance = 25%
    (mentalFactor) * 20 + // mental state = 20%
    (wellness / 100) * 15 + // wellness = 15%
    (parentFactor) * 5; // parent education = 5%

  return Math.min(Math.max(potential, 0), 100).toFixed(1);
}

// === BUTTON EVENT LISTENER ===
document.getElementById("calcBtn").addEventListener("click", () => {
  const data = {
    studyHours: document.getElementById("studyHours").value,
    attendance: document.getElementById("attendance").value,
    sleepHours: document.getElementById("sleepHours").value,
    socialMediaHours: document.getElementById("socialMediaHours").value,
    mentalState: document.getElementById("mentalState").value,
    parentEdu: document.getElementById("parentEdu").value
  };

  const score = calculateGradePotential(data);

  window.location.href = `score-summary.html?score=${score}`;
});

