// Debug script to test progress indicator
console.log('Testing progress indicator elements...');

// Test element selection
const progressCircle = document.getElementById('progress-circle');
const progressPercentage = document.getElementById('progress-percentage');
const progressStatus = document.getElementById('progress-status');
const stepIndicators = document.querySelectorAll('[id^="step-"]');

console.log('Progress Circle:', progressCircle);
console.log('Progress Percentage:', progressPercentage);
console.log('Progress Status:', progressStatus);
console.log('Step Indicators:', stepIndicators.length, stepIndicators);

// Test progress simulation
if (progressCircle && progressPercentage && progressStatus) {
    console.log('All elements found! Testing progress animation...');
    
    // Test direct progress update
    function testProgress() {
        const testProgress = 50;
        const circumference = 251.33;
        const strokeDashoffset = circumference - (testProgress / 100) * circumference;
        
        progressCircle.style.strokeDashoffset = strokeDashoffset;
        progressPercentage.textContent = testProgress + '%';
        progressStatus.textContent = 'Testing progress...';
        
        console.log('Set progress to 50%');
        console.log('Stroke dash offset:', strokeDashoffset);
    }
    
    // Run test after a delay
    setTimeout(testProgress, 2000);
} else {
    console.error('Some progress elements are missing!');
}
