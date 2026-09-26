// Typewriter effect for elements with class "typewriter"
// Usage in markdown (inside an HTML block):
// <span class="typewriter" data-text="npm install your-package"></span>

document.addEventListener("DOMContentLoaded", () => {
  const targets = document.querySelectorAll(".typewriter");

  targets.forEach((el) => {
    const text = el.getAttribute("data-text") || "";
    const speed = parseInt(el.getAttribute("data-speed") || "45", 10);
    el.textContent = "";
    let i = 0;

    function type() {
      if (i <= text.length) {
        el.textContent = text.slice(0, i);
        i++;
        setTimeout(type, speed);
      }
    }
    type();
  });
});
