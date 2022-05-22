// init jquery
(() => { window.$ = window.jQuery = require("jquery"); })()

const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const knnClassifier = require('@tensorflow-models/knn-classifier');

const $videoInput = $('#videoInput');
const $previewVideo = $('#previewVideo');
const $statVideo = $('#statVideo');
const videoElem = $previewVideo[0];
const statVideoElem = $statVideo[0];

const canvas = $('#canvas')[0];
const ctx = canvas.getContext('2d');

const statCanvas = $('#statCanvas')[0];
const statCtx = statCanvas.getContext('2d');

const scopeCanvas = $('#scopeCanvas')[0];
const scopeCtx = scopeCanvas.getContext('2d');

const rect = { x: 257, y: 7, width: 440, height: 1060 }
let upSeconds = 0;
const $upSeconds = $('#up-seconds');

let midSeconds = 0;
const $midSeconds = $('#mid-seconds');

let downSeconds = 0;
const $downSeconds = $('#down-seconds');

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

let locked = false;

const redraw = () => {
    if (!locked) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(videoElem, 0, 0);
        ctx.beginPath();
        ctx.strokeStyle = "red";
        ctx.lineWidth = 5;
        ctx.rect(rect.x, rect.y, rect.width, rect.height);
        ctx.stroke();
    }

    statCtx.clearRect(0, 0, canvas.width, canvas.height);
    statCtx.drawImage(statVideoElem, 0, 0);

    scopeCanvas.width = rect.width;
    scopeCanvas.height = rect.height;
    const imageData = statCtx.getImageData(rect.x, rect.y, rect.width, rect.height);
    scopeCtx.clearRect(0, 0, scopeCanvas.width, scopeCanvas.height);
    scopeCtx.putImageData(imageData, 0, 0); 

    window.requestAnimationFrame(redraw);
}
window.requestAnimationFrame(redraw);

$videoInput.on('change', () => {
    const videoFile = $videoInput[0].files[0];
    const videoSrc = URL.createObjectURL(videoFile);
    $previewVideo.attr('src', videoSrc);
    $statVideo.attr('src', videoSrc);
});

$previewVideo.on('loadedmetadata', function() {
    canvas.width = this.videoWidth;
    canvas.height = this.videoHeight;
    statCanvas.width = this.videoWidth;
    statCanvas.height = this.videoHeight;
});

$('#x, #y, #width, #height').on('change', function() {
    const $input = $(this);
    const attrName = $input.attr('id');
    rect[attrName] = Math.max(Number($input.val()), 0);
});

canvas.addEventListener('mousedown', function(event) {
    if (locked) return;

    const canvasRect = canvas.getBoundingClientRect();
    rect.x = Math.floor(event.clientX - canvasRect.left);
    rect.y = Math.floor(event.clientY - canvasRect.top);
    $('#x').val(rect.x);
    $('#y').val(rect.y);
});

$('#lock').on('change', function() {
    locked = $(this).prop('checked');
    if (locked) {
        $('#x, #y, #width, #height').prop('readonly', true);
    } else {
        $('#x, #y, #width, #height').prop('readonly', false);
    }
})


let classifier = null;
let mobilenetModule = null;

(async function() {
    // Create the classifier.
    classifier = knnClassifier.create();

    // Load mobilenet.
    mobilenetModule = await mobilenet.load();

    // Add MobileNet activations to the model repeatedly for all classes.
    $('.up').each(function(idx, img) {
        console.log('load up:', idx + 1);
        const img0 = tf.browser.fromPixels(img);
        const logits0 = mobilenetModule.infer(img0, true);
        classifier.addExample(logits0, 0);
    })
    $('.mid').each(function(idx, img) {
        console.log('load mid:', idx + 1);
        const img1 = tf.browser.fromPixels(img);
        const logits1 = mobilenetModule.infer(img1, true);
        classifier.addExample(logits1, 1);
    })
    $('.down').each(function(idx, img) {
        console.log('load down:', idx + 1);
        const img2 = tf.browser.fromPixels(img);
        const logits2 = mobilenetModule.infer(img2, true);
        classifier.addExample(logits2, 2);
    })
    alert('data loaded');
})();

const predict = async () => {
    const x = statCtx.getImageData(rect.x, rect.y, rect.width, rect.height);
    const xlogits = mobilenetModule.infer(x, true);
    return await classifier.predictClass(xlogits);
}

let pasue = false;
$('#pauseStat').on('click', () => {
    $(this).prop('disabled', true);
    $('#startStat').prop('disabled', false);
    statVideoElem.pause();
    pause = true;
});

const roundNum = (num) => Math.round(num * 100) / 100;
$('#startStat').on('click', function () {
    $(this).prop('disabled', true);
    $('#pauseStat').prop('disabled', false);
    pause = false;

    (async () => {
        const duration = statVideoElem.duration;
        let lastCurrentTime = statVideoElem.currentTime;
        let currentTime = statVideoElem.currentTime;
        statVideoElem.play();

        while(!pause && (currentTime + 0.1 < duration)) {
            currentTime = statVideoElem.currentTime;
            const changedTime = currentTime - lastCurrentTime;
            lastCurrentTime = currentTime;

            const result = await predict()
            if (result.classIndex === 0) {
                upSeconds += changedTime;
                $upSeconds.text(roundNum(upSeconds));
            } else if (result.classIndex === 1) {
                midSeconds += changedTime;
                $midSeconds.text(roundNum(midSeconds));
            } else if (result.classIndex === 2) {
                downSeconds += changedTime;
                $downSeconds.text(roundNum(downSeconds));
            }
        }        
    })();
});

$('#resetStat').on('click', function () {
    upSeconds = 0.0;
    $upSeconds.text(roundNum(upSeconds));
    midSeconds = 0.0;
    $midSeconds.text(roundNum(midSeconds));
    downSeconds = 0.0;
    $downSeconds.text(roundNum(downSeconds));
});