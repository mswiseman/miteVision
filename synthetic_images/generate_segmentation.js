// generate_segmentation.js
//
// Generate random images with segmentation masks given a list of classes, segmented objects from those classes, and
// background images
//
// Author: Michele Wiseman
// Date 2/21/2025
// Version 1.0


const fs = require('fs');
const path = require('path');
const os = require('os');

const async = require('async');
const _ = require('lodash');

const { createCanvas, loadImage } = require('canvas');

const OUTPUT_DIR = path.join(__dirname, "Output");
const MASK_OUTPUT_DIR = path.join(__dirname, "Masks");
const BACKGROUND_IMAGES = path.join(__dirname, "Background_Images");
const SEGMENTED_IMAGES = path.join(__dirname, "Training");

const IMAGES_TO_GENERATE = 100;
const CONCURRENCY = Math.max(1, os.cpus().length - 1);
const CANVAS_WIDTH = 8254;
const CANVAS_HEIGHT = 5502;
const MAX_OBJECTS = 50;

// Define unique colors for classes
const CLASS_COLORS = [
    [255, 0, 0],    // Red
    [0, 255, 0],    // Green
    [0, 0, 255],    // Blue
    [255, 255, 0],  // Yellow
    [255, 0, 255],  // Magenta
];

// Create output directories
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR);
if (!fs.existsSync(MASK_OUTPUT_DIR)) fs.mkdirSync(MASK_OUTPUT_DIR);

// Create class-specific mask directories
const CLASS_MASK_DIRS = {};
CLASS_COLORS.forEach((_, classId) => {
    const classDir = path.join(MASK_OUTPUT_DIR, `masks_${classId}`);
    if (!fs.existsSync(classDir)) fs.mkdirSync(classDir);
    CLASS_MASK_DIRS[classId] = classDir;
});

// Load classes and object images
const folders = _.filter(fs.readdirSync(SEGMENTED_IMAGES), function (filename) {
    const fullPath = path.join(SEGMENTED_IMAGES, filename);
    return fs.statSync(fullPath).isDirectory();
});

const OBJECTS = {};
_.each(folders, function (folder) {
    const classObjects = _.filter(fs.readdirSync(path.join(SEGMENTED_IMAGES, folder)), function (filename) {
        return filename.match(/\.png$/i);
    }).map(image => path.join(folder, image));

    OBJECTS[folder] = classObjects;
});

// Generate random images with segmentation masks
const createImage = function (filename, cb) {
    const BG = _.sample(fs.readdirSync(BACKGROUND_IMAGES).filter(file => file.match(/\.(jpg|jpeg|png)$/i)));

    if (!BG) {
        console.error("No valid background found");
        return cb(null);
    }

    loadImage(path.join(BACKGROUND_IMAGES, BG)).then(bgImage => {
        const canvas = createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
        const context = canvas.getContext('2d');
        const maskCanvas = createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
        const maskContext = maskCanvas.getContext('2d');

        // Fill mask background with black
        maskContext.fillStyle = "black";
        maskContext.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

        // Draw the background on the main canvas
        const scale = Math.max(CANVAS_WIDTH / bgImage.width, CANVAS_HEIGHT / bgImage.height);
        const x = (CANVAS_WIDTH - bgImage.width * scale) / 2;
        const y = (CANVAS_HEIGHT - bgImage.height * scale) / 2;
        context.drawImage(bgImage, x, y, bgImage.width * scale, bgImage.height * scale);

        const objects = 1 + Math.floor(Math.random() * Math.random() * (MAX_OBJECTS - 1));

        async.timesSeries(objects, function (i, cb) {
            addRandomObject(canvas, context, maskCanvas, maskContext, filename, i, cb);
        }, function () {
            // Save the copy-paste composite image
            const imageOut = fs.createWriteStream(path.join(OUTPUT_DIR, `${filename}.jpg`));
            const imageStream = canvas.createJPEGStream({ quality: 0.95 });
            imageStream.pipe(imageOut);

            imageOut.on('finish', () => {
                console.log(`Copy-paste composite image saved: ${filename}.jpg`);

                // Save the combined mask
                const maskOut = fs.createWriteStream(path.join(MASK_OUTPUT_DIR, `${filename}_combined.png`));
                const maskStream = maskCanvas.createPNGStream();
                maskStream.pipe(maskOut);

                maskOut.on('finish', () => {
                    console.log(`Combined mask saved: ${filename}_combined.png`);
                    cb(null);
                });
            });
        });
    }).catch(err => {
        console.error("Error loading background image:", err);
        cb(null);
    });
};


const addRandomObject = function (canvas, context, maskCanvas, maskContext, filename, objectIndex, cb) {
    const className = _.sample(Object.keys(OBJECTS));
    const classId = Object.keys(OBJECTS).indexOf(className); // Get class ID
    const objectPath = _.sample(OBJECTS[className]);

    if (!objectPath) {
        console.error(`No objects found for class: ${className}`);
        return cb(null);
    }

    loadImage(path.join(SEGMENTED_IMAGES, objectPath)).then(objImage => {
        const objCanvas = createCanvas(objImage.width, objImage.height);
        const objContext = objCanvas.getContext('2d');
        objContext.drawImage(objImage, 0, 0, objImage.width, objImage.height);

        const scale = 1 + Math.random() * 0.2 - 0.1;
        const objWidth = objImage.width * scale;
        const objHeight = objImage.height * scale;

        const x = Math.random() * (CANVAS_WIDTH - objWidth);
        const y = Math.random() * (CANVAS_HEIGHT - objHeight);

        const radians = Math.random() * Math.PI * 2;

        // Draw object on the main canvas
        context.save();
        context.translate(x + objWidth / 2, y + objHeight / 2);
        context.rotate(radians);
        context.drawImage(objCanvas, -objWidth / 2, -objHeight / 2, objWidth, objHeight);
        context.restore();

        // Draw object on the combined mask canvas with its class color
        maskContext.save();
        maskContext.translate(x + objWidth / 2, y + objHeight / 2);
        maskContext.rotate(radians);
        const [r, g, b] = CLASS_COLORS[classId];
        maskContext.globalCompositeOperation = "source-over";

        // Process transparency to apply class color
        const objImageData = objContext.getImageData(0, 0, objImage.width, objImage.height);
        const data = objImageData.data;

        for (let i = 0; i < data.length; i += 4) {
            const alpha = data[i + 3]; // Alpha channel
            if (alpha > 0) {
                data[i] = r;     // Red
                data[i + 1] = g; // Green
                data[i + 2] = b; // Blue
            } else {
                // Set fully transparent pixels to black for masks
                data[i] = 0;
                data[i + 1] = 0;
                data[i + 2] = 0;
                data[i + 3] = 255; // Fully opaque black
            }
        }

        objContext.putImageData(objImageData, 0, 0);
        maskContext.drawImage(objCanvas, -objWidth / 2, -objHeight / 2, objWidth, objHeight);
        maskContext.restore();

        // Save a separate class-specific mask
        const classMaskCanvas = createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
        const classMaskContext = classMaskCanvas.getContext('2d');

        classMaskContext.fillStyle = "black";
        classMaskContext.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

        classMaskContext.save();
        classMaskContext.translate(x + objWidth / 2, y + objHeight / 2);
        classMaskContext.rotate(radians);
        classMaskContext.globalCompositeOperation = "source-over";
        classMaskContext.drawImage(objCanvas, -objWidth / 2, -objHeight / 2, objWidth, objHeight);
        classMaskContext.restore();

        const classMaskPath = path.join(CLASS_MASK_DIRS[classId], `${filename}_object_${objectIndex}.png`);
        const maskOut = fs.createWriteStream(classMaskPath);
        const maskStream = classMaskCanvas.createPNGStream();
        maskStream.pipe(maskOut);

        maskOut.on('finish', () => {
            console.log(`Class-specific mask saved: ${classMaskPath}`);
            cb(null);
        });
    }).catch(err => {
        console.error("Error loading object image:", err);
        cb(null);
    });
};


// Generate images
async.timesLimit(IMAGES_TO_GENERATE, CONCURRENCY, function (i, cb) {
    createImage(`image_${i}`, cb);
}, function () {
    console.log("All images and masks generated.");
});
