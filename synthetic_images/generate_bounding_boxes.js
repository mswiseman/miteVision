// node generate_bounding_boxes.js

// system packages
const fs = require('fs');
const path = require('path');
const os = require('os');

// basic helpers
const async = require('async');
const _ = require('lodash');

// drawing utilities
const { createCanvas, loadImage, CanvasRenderingContext2D } = require('canvas');
const floodfill = require('@roboflow/floodfill')(CanvasRenderingContext2D);

// for writing annotations
var Handlebars = require('handlebars');
var vocTemplate = Handlebars.compile(fs.readFileSync(__dirname + "/voc.tmpl", "utf-8"));

// how many images we want to create
const IMAGES_TO_GENERATE = 30;
// how many to generate at one time
const CONCURRENCY = Math.max(1, os.cpus().length - 1);

// approximate aspect ratio of our phone camera
// scaled to match the input of CreateML models
const CANVAS_WIDTH = 8254;
const CANVAS_HEIGHT = 5502;

// the most objects you want in your generated images
const MAX_OBJECTS = 50;

// where to store our images
const OUTPUT_DIR = path.join(__dirname, "Output");

// location of jpgs on your filesystem
const BACKGROUND_IMAGES = path.join(__dirname, "Background_Images");
// text file images
const BACKGROUNDS = fs.readFileSync(__dirname + "/Background_Images.filtered.txt", "utf-8").split("\n");

// location of folders containing jpgs)
const SEGMENTED_IMAGES = path.join(__dirname, "Training");

// get class names
const folders = _.filter(fs.readdirSync(SEGMENTED_IMAGES), function(filename) {
    // filter out hidden files like .DS_STORE
    return filename.indexOf('.') != 0;
});
var classes = _.map(folders, function(folder) {
    // This dataset has some classes like "Apple Golden 1" and "Apple Golden 2"
    // We want to combine these into just "Apple" so we only take the first word
    return folder.split(" ")[0];
});

// for each class, get a list of images
const OBJECTS = {};
_.each(folders, function(folder, i) {
    var cls = classes[i]; // get the class name

    var objs = [];
    objs = _.filter(fs.readdirSync(path.join(SEGMENTED_IMAGES, folder)), function(filename) {
        // Update this line to match .png files or both .jpg and .png
        return filename.match(/\.png$/); // or use /\.(jpe?g|png)$/ to match both
    });

    objs = _.map(objs, function(image) {
        // we need to know which folder this came from
        return path.join(folder, image);
    });

    if(!OBJECTS[cls]) {
        OBJECTS[cls] = objs;
    } else {
        // append to existing images
        _.each(objs, function(obj) {
            OBJECTS[cls].push(obj);
        });
    }
});

// Example class weights - the higher the number, the more likely the class is to be selected
const classWeights = {
    'Immature': 0.3, // weight
    'Viable_egg': 0.1, // weight
    'Adult_female': 0.9, // weight
    'Dead_mite': 0.3, // weight
    'Adult_male': 1//
  // ... add weights for all classes
};

// Create a weighted list based on the class weights
let weightedClasses = [];
for (const [className, weight] of Object.entries(classWeights)) {
  for (let i = 0; i < weight; i++) {
    weightedClasses.push(className);
  }
}

// Now weightedClasses has a weighted distribution of class names, e.g., ['classA', 'classB', 'classB', 'classB', 'classC', 'classC']

// Randomly select a class from the weighted list
const randomIndex = Math.floor(Math.random() * weightedClasses.length);
const selectedClass = weightedClasses[randomIndex];

// when we randomly select a class, we want them equally weighted
classes = _.uniq(classes);

// create our output directory if it doesn't exist
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR);

// create the images
_.defer(function() {
    var num_completed = 0;
    const progress_threshold = Math.max(1, Math.round( Math.min(100, IMAGES_TO_GENERATE/1000) ) );
    async.timesLimit(IMAGES_TO_GENERATE, CONCURRENCY, function(i, cb) {
        createImage(i, function() {
            // record progress to console
            num_completed++;
            if(num_completed%progress_threshold === 0) {
                console.log((num_completed/IMAGES_TO_GENERATE*100).toFixed(1)+'% finished.');
            }
            cb(null);
        });
    }, function() {
        // completely done generating!
        console.log("Done");
        process.exit(0);
    });
});

const createImage = function(filename, cb) {
    // select and load a random background
    const BG = _.sample(BACKGROUNDS);

    // Check if BG is undefined or not a string
    if (typeof BG !== 'string') {
        console.error('Invalid background path:', BG);
        return cb(null); // Skip this iteration
    }

    loadImage(path.join(BACKGROUND_IMAGES, BG)).then(function(img) {
        var canvas = createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
        var context = canvas.getContext('2d');

        // scale the background to fill our canvas and paint it in the center
        var scale = Math.max(canvas.width / img.width, canvas.height / img.height);
        var x = (canvas.width / 2) - (img.width / 2) * scale;
        var y = (canvas.height / 2) - (img.height / 2) * scale;
        context.drawImage(img, x, y, img.width * scale, img.height * scale);

        // calculate how many objects to add
        // highest probability is 1, then 2, then 3, etc up to MAX_OBJECTS
        // if you want a uniform probability, remove one of the Math.random()s
        var objects = 1+Math.floor(Math.random()*Math.random()*(MAX_OBJECTS-1));

        var boxes = [];
        async.timesSeries(objects, function(i, cb) {
            // for each object, add it to the image and then record its bounding box
            addRandomObject(canvas, context, function(box) {
                boxes.push(box); // record the bounding box
                cb(null); //
            });
        }, function() {
            // write our files to disk
            async.parallel([
                function(cb) {
                    // Write the JPG file with higher quality
                    const out = fs.createWriteStream(path.join(__dirname, "Output", filename+".jpg"));
                    const stream = canvas.createJPEGStream({
                        quality: 0.95 // Set JPEG quality to 95%
                    });
                    stream.pipe(out);
                    out.on('finish', function() {
                        cb(null);
                    });
                },
                function(cb) {
                    // write the bounding boxes to the XML annotation file
                    fs.writeFileSync(
                        path.join(__dirname, "Output", filename+".xml"),
                        vocTemplate({
                            filename: filename + ".jpg",
                            width: CANVAS_WIDTH,
                            height: CANVAS_HEIGHT,
                            boxes: boxes
                        })
                    );

                    cb(null);
                }
            ], function() {
                // we're done generating this image
                cb(null);
            });
        });
    });
};

const addRandomObject = function(canvas, context, cb) {
    // Select a class using the weighted class list
    const randomIndex = Math.floor(Math.random() * weightedClasses.length);
    const cls = weightedClasses[randomIndex];

    const object = _.sample(OBJECTS[cls]);

    // Check if object is undefined or not a string
    if (typeof object !== 'string') {
        console.error('Invalid object path:', object);
        return cb(null); // Skip this iteration
    }

    loadImage(path.join(SEGMENTED_IMAGES, object)).then(function(img) {
        // erase white edges
        var objectCanvas = createCanvas(img.width, img.height);
        var objectContext = objectCanvas.getContext('2d');

        objectContext.drawImage(img, 0, 0, img.width, img.height);

        // flood fill starting at all the corners
        const tolerance = 10;
        objectContext.fillStyle = "rgba(0,255,0,0)";
        objectContext.fillFlood(3, 0, tolerance); // top left
        objectContext.fillFlood(img.width-1, 0, tolerance); // top right
        objectContext.fillFlood(img.width-1, img.height-1, tolerance); // bottom right
        objectContext.fillFlood(0, img.height-1, tolerance); // bottom left

        // cleanup edges
        objectContext.blurEdges(0.15); // blur the edges

        // make them not all look exactly the same
        objectContext.randomHSL(0.00, 0.1, 0.1); // change the hue, saturation, and lightness

        // randomly scale the image
        var baseScale = 1; // Base scale increased by 20%
        var scaleVariation = 0.02; // Variation of scale
        const scale = baseScale + Math.random() * scaleVariation * 2 - scaleVariation; // Randomly scale

        var w = img.width * scale; // width of the object
        var h = img.height * scale; // height of the object

        // place object at random position on top of the background
        const max_width = canvas.width - w; // max width of the object
        const max_height = canvas.height - h; // max height of the object

        var x = Math.floor(Math.random()*max_width); // random x position
        var y = Math.floor(Math.random()*max_height); // random y position

        context.save();

        // randomly rotate and draw the image
        const radians = Math.random() * Math.PI * 2; // random rotation
        context.translate(x + w / 2, y + h / 2); // move to the center of the object
        context.rotate(radians); // rotate the object

        // draw the object
        context.drawImage(objectCanvas, -w / 2, -h / 2, w, h);

        context.restore(); // restore the context to its original state

        // calculate the new bounding box after the rotation
        const cos = Math.cos(radians);
        const sin = Math.sin(radians);
        const halfWidth = w / 2;
        const halfHeight = h / 2;
        const corners = [
            {x: -halfWidth, y: -halfHeight},
            {x: +halfWidth, y: -halfHeight},
            {x: +halfWidth, y: +halfHeight},
            {x: -halfWidth, y: +halfHeight}
        ];
        const rotatedCorners = corners.map(corner => {
            return {
                x: corner.x * cos - corner.y * sin,
                y: corner.x * sin + corner.y * cos
            };
        });
        const minX = Math.min(...rotatedCorners.map(corner => corner.x));
        const maxX = Math.max(...rotatedCorners.map(corner => corner.x));
        const minY = Math.min(...rotatedCorners.map(corner => corner.y));
        const maxY = Math.max(...rotatedCorners.map(corner => corner.y));

        // return the type and bounds of the object we placed
        cb({
            cls: cls,
            xmin: Math.floor(x + w / 2 + minX),
            xmax: Math.ceil(x + w / 2 + maxX),
            ymin: Math.floor(y + h / 2 + minY),
            ymax: Math.ceil(y + h / 2 + maxY)
        });
    });
};
