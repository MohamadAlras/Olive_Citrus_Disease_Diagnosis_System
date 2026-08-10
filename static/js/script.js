document.addEventListener("DOMContentLoaded", function(){

    console.log("Olive Citrus Diagnosis System Loaded");

});

/* إظهار وإخفاء كلمة المرور */

document.addEventListener("DOMContentLoaded", () => {

    document.querySelectorAll(".toggle-password").forEach(button => {

        button.addEventListener("click", function () {

            const wrapper = this.closest(".password-wrapper");
            const input = wrapper.querySelector("input");

            if (input.type === "password") {
                input.type = "text";
                this.innerHTML = '<i class="bi bi-eye-slash"></i>';
            } else {
                input.type = "password";
                this.innerHTML = '<i class="bi bi-eye"></i>';
            }

        });

    });

});