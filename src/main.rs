#![feature(let_chains)]
#![feature(generic_arg_infer)]
use anyhow::*;
use enterpolation::{linear::ConstEquidistantLinear, Curve};
use minifb::{Key, Window, WindowOptions};
use palette::{IntoColor, LinSrgb, Srgb};
use serialport::SerialPort;
use std::io::{IoSliceMut, Read};
use std::result::Result::Ok;
use std::thread::sleep;
use std::time::Duration;
use std::{default, str};

use rand::Rng;
// use winit_input_helper::WinitInputHelper;

const FRAME_BEGIN_FLAG: u16 = 0xFF00;
const FRAME_END_FLAG: u8 = 0xDD;

const FRAME_HEAD_SIZE: usize = 20;

use zerocopy::{FromBytes, FromZeroes};

#[derive(FromZeroes, FromBytes, Debug)]
#[repr(C)]
struct FrameHead {
    frame_begin_flag: u16,
    frame_data_len: u16,
    reserved1: u8, // fixed to 0xff
    output_mode: u8,
    senser_temp: u8,
    driver_temp: u8,
    exposure_time: [u8; 4],
    error_code: u8,
    reserved2: u8, // fixed to 0x00
    resolution_rows: u8,
    resolution_cols: u8,
    frame_id: u16, // 12-bit, 0~4095
    isp_version: u8,
    reserved3: u8, // fixed to 0xff
}

fn draw_circle(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    radius: usize,
    color: u32,
) {
    for y in 0..height {
        for x in 0..width {
            let dx = x as isize - cx as isize;
            let dy = y as isize - cy as isize;
            if dx * dx + dy * dy <= (radius * radius) as isize {
                buffer[y * width + x] = color;
            }
        }
    }
}
fn main() -> Result<()> {
    let mut port = serialport::new("/dev/ttyUSB1", 115_200)
        .timeout(Duration::from_millis(10))
        .open()
        .expect("Failed to open port");

    port.write(b"AT+DISP=2\r").ok();
    sleep(Duration::from_millis(20));
    port.write(b"AT+FPS=19\r").ok();
    sleep(Duration::from_millis(20));
    port.write(b"AT+UNIT=3\r").ok();
    sleep(Duration::from_millis(20));
    port.write(b"AT+ANTIMMI=-1\r").ok();
    sleep(Duration::from_millis(20));
    port.write(b"AT+BAUD=3000000\r").ok();
    sleep(Duration::from_millis(20));

    let width = 100;
    let height = 100;
    let mut window = Window::new(
        "Depth Image",
        width,
        height,
        WindowOptions {
            scale: minifb::Scale::X4,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    let mut buf = [0u8; 10022];
    let mut output = [0u32; 100 * 100];
    let gradient = ConstEquidistantLinear::<f32, _, _>::equidistant_unchecked([
        LinSrgb::new(0.5, 0.0, 0.0),
        LinSrgb::new(1.0, 0.0, 0.0),
        LinSrgb::new(1.0, 1.0, 0.0),
        LinSrgb::new(0.0, 1.0, 0.0),
        LinSrgb::new(0.0, 1.0, 1.0),
        LinSrgb::new(0.0, 0.0, 1.0),
        LinSrgb::new(0.0, 0.0, 0.3),
    ]);
    let gradient: Vec<LinSrgb> = gradient.take(256).collect();
    let mut rng = rand::thread_rng();
    const RG: std::ops::Range<usize> = 3..97;
    let mut cp = (rng.gen_range(RG), rng.gen_range(RG));
    let mut count = 1;
    let mut count1 = 0;
    window.set_target_fps(60);
    while window.is_open() && !window.is_key_down(Key::Escape) {
        port.read_exact(&mut buf).ok();
        if let Some(header) = FrameHead::ref_from_prefix(&buf)
            && header.frame_begin_flag == 0xff00
            && buf.ends_with(&[0xDD])
        {
            let c = window.is_key_pressed(Key::C, minifb::KeyRepeat::No);
            let n = window.is_key_pressed(Key::N, minifb::KeyRepeat::No);
            if n {
                println!("count: {count} count1: {count1}");
                // let filename = format!("training_data/depth_image_{}", cp.0, cp.1);
                std::fs::write(
                    format!("training_data/d_{}_{}", count, count1),
                    &buf[20..(20 + 10000)],
                )
                .ok();
                count1 = count1 + 1;
                // cp = (rng.gen_range(RG), rng.gen_range(RG));
            }
            if c {
                count = count + 1;
                count1 = 0;
            }
            for (i, pixel) in buf[20..].iter().take(width * height).enumerate() {
                let intensity = 255 - *pixel as usize;
                let color = gradient[intensity].into_format::<u8>();
                output[i] =
                    (color.red as u32) << 16 | (color.green as u32) << 8 | color.blue as u32;
            }

            // draw_circle(&mut output, 100, 100, cp.0, cp.1, 3, 0xFFFFFF);

            // window.get_mouse_pos(minifb::MouseMode::Clamp).map(|mouse| {
                // println!(
                //     "x {} y {} value: {}",
                //     mouse.0,
                //     mouse.1,
                //     255 - buf[(mouse.1 as usize * 100) + mouse.0 as usize + 22]
                // );
            }); // window.draw_circle(cp.0 as i32, cp.1 as i32, 10, 0xFF0000);
            window.update_with_buffer(&output, width, height).unwrap();
            // dbg!(&buf[10015]);
            // dbg!(header);
        }
    }

    Ok(())
}
