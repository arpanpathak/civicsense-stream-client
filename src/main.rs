// ============================================================================
// Pistream client – YOLOv8 object detection on an MJPEG stream, using candle
// ============================================================================

mod coco_classes;
mod model;

use anyhow::Result;
use candle_core::{Device, DType, IndexOp, Tensor};
use candle_nn::Module;
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use futures_util::StreamExt;
use image::Rgb;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use model::YoloV8;
use reqwest::Client;
use std::path::Path;
use std::sync::Arc;

type JpegFrame = Vec<u8>;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Pistream Client (candle YOLO) starting...");

    // Load model
    let model_path = "models/yolov8n.safetensors";
    if !Path::new(model_path).exists() {
        eprintln!("❌ Model file not found: {}", model_path);
        eprintln!("   Download: curl -L -o models/yolov8n.safetensors https://huggingface.co/lmz/candle-yolo-v8/resolve/main/yolov8n.safetensors");
        std::process::exit(1);
    }

    let device = Device::Cpu;
    let model = YoloV8::load_from_file(model_path, &device)?;
    println!("✅ Model loaded");

    // Output dir
    let output_dir = "output";
    std::fs::create_dir_all(output_dir)?;
    println!("📁 Saving to {}", output_dir);

    // Stream
    let stream_url = std::env::var("STREAM_URL")
        .unwrap_or_else(|_| "http://192.168.0.43:8000".to_string());
    println!("🌐 Connecting to {}", stream_url);
    let client = Client::new();
    let response = client.get(stream_url).send().await?;
    if !response.status().is_success() {
        anyhow::bail!("HTTP error: {}", response.status());
    }

    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    let mut frame_counter = 0u32;
    let model = Arc::new(model);

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        buffer.extend_from_slice(&chunk);

        while let Some(jpeg) = extract_jpeg(&mut buffer) {
            frame_counter += 1;
            if frame_counter % 30 == 0 {
                println!("📸 Received {} frames", frame_counter);
            }

            let annotated = run_detection(&jpeg, &model)?;
            let filename = format!("{}/frame_{:06}.jpg", output_dir, frame_counter);
            std::fs::write(&filename, annotated)?;
        }
    }
    Ok(())
}

fn extract_jpeg(buffer: &mut Vec<u8>) -> Option<JpegFrame> {
    let start = buffer.windows(2).position(|w| w == [0xFF, 0xD8])?;
    let end = buffer[start..].windows(2).position(|w| w == [0xFF, 0xD9])?;
    let end_pos = start + end + 2;
    let frame = buffer[start..end_pos].to_vec();
    buffer.drain(..end_pos);
    Some(frame)
}

fn run_detection(jpeg_data: &[u8], model: &YoloV8) -> Result<JpegFrame> {
    let img = image::load_from_memory(jpeg_data)?;
    let (orig_w, orig_h) = (img.width(), img.height());

    // Resize so that the longest side is 640 while keeping the aspect ratio.
    // The other side is rounded down to a multiple of 32 (YOLO stride).
    let (width, height) = {
        let w = img.width() as usize;
        let h = img.height() as usize;
        if w < h {
            let w = w * 640 / h;
            (w / 32 * 32, 640)
        } else {
            let h = h * 640 / w;
            (640, h / 32 * 32)
        }
    };
    let image_t = {
        let img = img.resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let data = img.to_rgb8().into_raw();
        Tensor::from_vec(
            data,
            (img.height() as usize, img.width() as usize, 3),
            &Device::Cpu,
        )?
        .permute((2, 0, 1))?
    };
    let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
    let predictions = model.forward(&image_t)?.squeeze(0)?;

    // Group the raw predictions into per-class bounding boxes.
    let (pred_size, npreds) = predictions.dims2()?;
    let nclasses = pred_size - 4;
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(predictions.i((.., index))?)?;
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > 0.45 {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }
    non_maximum_suppression(&mut bboxes, 0.6);

    // Draw the surviving boxes on the original-size image.
    let w_ratio = orig_w as f32 / width as f32;
    let h_ratio = orig_h as f32 / height as f32;
    let mut annotated = img.to_rgb8();
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!(
                "  {}: {:.0}% at ({:.0},{:.0})-({:.0},{:.0})",
                coco_classes::NAMES[class_index],
                100. * b.confidence,
                b.xmin,
                b.ymin,
                b.xmax,
                b.ymax
            );
            let xmin = (b.xmin * w_ratio) as i32;
            let ymin = (b.ymin * h_ratio) as i32;
            let dx = (b.xmax - b.xmin) * w_ratio;
            let dy = (b.ymax - b.ymin) * h_ratio;
            if dx >= 0. && dy >= 0. {
                draw_hollow_rect_mut(
                    &mut annotated,
                    Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                    Rgb([255, 0, 0]),
                );
            }
        }
    }

    let mut out = Vec::new();
    annotated.write_to(&mut std::io::Cursor::new(&mut out), image::ImageOutputFormat::Jpeg(90))?;
    Ok(out)
}
