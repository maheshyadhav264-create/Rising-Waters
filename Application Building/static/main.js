// main.js
// Adds simple client-side validation and a submit-loading state
// to the flood prediction form.

document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('floodForm');
  if (!form) return;

  const errorEl = document.getElementById('formError');
  const submitBtn = form.querySelector('button[type="submit"]');

  form.addEventListener('submit', function (event) {
    const inputs = form.querySelectorAll('input[required]');
    let firstInvalid = null;

    inputs.forEach(function (input) {
      const value = parseFloat(input.value);
      const isEmpty = input.value.trim() === '';
      const isNegative = !isNaN(value) && value < 0;
      const overHundred = input.id === 'cloud_cover' && value > 100;

      const invalid = isEmpty || isNaN(value) || isNegative || overHundred;
      input.classList.toggle('invalid', invalid);

      if (invalid && !firstInvalid) firstInvalid = input;
    });

    if (firstInvalid) {
      event.preventDefault();
      errorEl.textContent = 'Please enter valid, non-negative values for every field (cloud cover must be 0–100).';
      firstInvalid.focus();
      return;
    }

    errorEl.textContent = '';
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Predicting…';
    }
  });
});
