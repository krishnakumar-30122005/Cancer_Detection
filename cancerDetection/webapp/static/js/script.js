// static/js/script.js
document.getElementById("upload-form").addEventListener("submit", function(e) {
    e.preventDefault();

    const formData = new FormData();
    const file = document.getElementById("file").files[0];
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            document.getElementById("cancer-type").innerText = data.cancer_type;
            document.getElementById("diagnosis").innerText = data.result;
            document.getElementById("result").classList.remove("hidden");
        }
    })
    .catch(error => {
        alert("Prediction failed.");
        console.error(error);
    });
});
