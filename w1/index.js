let net;


async function app() {
  console.log('Loading mobilenet..');
  net = await mobilenet.load();
  // Load the model.
  console.log('Sucessfully loaded model');
  document.getElementById("predictButton").onclick = async function () {
    var imgEl = document.getElementById("blah");
    const classifier = await net.classify(imgEl);
    console.log(classifier);
    var resultClass = document.getElementById("result");
    resultClass.innerHTML = classifier[0].className

  }
}


function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      var imgEl = document.getElementById("blah");
      var newImage = e.target.result;
      imgEl.setAttribute("src", newImage);
    }

    reader.readAsDataURL(input.files[0]);
  }
}



app();