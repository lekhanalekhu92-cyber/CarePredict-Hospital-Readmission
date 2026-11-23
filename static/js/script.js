document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");

    if (!form) return;

    form.addEventListener("submit", (event) => {
        const numberFields = form.querySelectorAll('input[type="number"]');

        for (const field of numberFields) {
            const value = Number(field.value);
            const min = Number(field.min);
            const max = Number(field.max);

            if (
                field.value.trim() === "" ||
                !Number.isFinite(value) ||
                value < min ||
                value > max
            ) {
                event.preventDefault();
                alert("Please enter values within the allowed ranges.");
                field.focus();
                return;
            }
        }
    });
});