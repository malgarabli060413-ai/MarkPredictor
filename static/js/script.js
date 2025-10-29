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
const calcBtn = document.getElementById("calcBtn");
if (calcBtn) {
  calcBtn.addEventListener("click", async () => {
  // Get form data - all fields that match the model
  const data = {
    age: document.getElementById("age").value,
    gender: document.getElementById("gender").value,
    studyHours: document.getElementById("studyHours").value,
    socialMediaHours: document.getElementById("socialMediaHours").value,
    netflixHours: document.getElementById("netflixHours").value,
    partTimeJob: document.getElementById("partTimeJob").value,
    attendance: document.getElementById("attendance").value,
    sleepHours: document.getElementById("sleepHours").value,
    dietQuality: document.getElementById("dietQuality").value,
    exerciseFrequency: document.getElementById("exerciseFrequency").value,
    parentEdu: document.getElementById("parentEdu").value,
    internetQuality: document.getElementById("internetQuality").value,
    mentalHealthRating: document.getElementById("mentalHealthRating").value,
    extracurricular: document.getElementById("extracurricular").value
  };

  // Validate required inputs
  const requiredFields = ['age', 'gender', 'studyHours', 'attendance', 'sleepHours', 
                         'socialMediaHours', 'netflixHours', 'exerciseFrequency', 
                         'mentalHealthRating'];
  
  for (let field of requiredFields) {
    if (!data[field] || data[field] === '' || 
        (typeof data[field] === 'string' && data[field].startsWith('Select'))) {
      alert(`Please fill in all required fields. Missing: ${field}`);
      return;
    }
  }

  try {
    // Show loading state
    const btn = document.getElementById("calcBtn");
    btn.disabled = true;
    btn.textContent = "Calculating...";

    // Send prediction request to Flask API
    const response = await fetch('http://localhost:5001/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });

    const result = await response.json();

    if (result.success) {
      // Store the score in sessionStorage for the summary page
      sessionStorage.setItem('predictedScore', result.predicted_score);
      sessionStorage.setItem('performanceLevel', result.performance_level);
      
      // Redirect to score summary page
      window.location.href = `score-summary.html?score=${result.predicted_score}`;
    } else {
      alert(`Error: ${result.message}`);
    }
  } catch (error) {
    alert("An error occurred while calculating your score. Please try again.");
    console.error('Error:', error);
  } finally {
    // Reset button state
    const btn = document.getElementById("calcBtn");
    btn.disabled = false;
    btn.textContent = "Calculate Your Potential";
  }
});
}

