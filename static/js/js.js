function toggleSection(header) {
            const content = header.nextElementSibling;
            const icon = header.querySelector('.toggle-icon');
            
            content.classList.toggle('active');
            icon.classList.toggle('active');
        }


function updateMedia(value) {
    document.getElementById("media").textContent = value;

    fetch("/act_media", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ "dato": value })
    })
    .then(r => r.json())
    .then(data => console.log("Media actualizada:", data));
}

function updateDesvEst(value) {
    document.getElementById("desvEst").textContent = value;

    fetch("/act_desv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ "dato": value })
    }).then(r => r.json())
    .then(data => console.log("desvEst actualizada:", data));;

}

function updateTam(valor) {
    value = (valor*2)+1
    console.log(value)
    document.getElementById("tam").textContent = value;

    fetch("/act_tam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ "dato": value })
    }).then(r => r.json())
    .then(data => console.log("tam actualizada:", data));;

}

function updateVarianza(value) {
    document.getElementById("varianza").textContent = value;

    fetch("/act_varianza", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ "dato": value })
    }).then(r => r.json())
    .then(data => console.log("varianza actualizada:", data));;

}
