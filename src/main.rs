#![feature(let_chains)]
#![feature(generic_arg_infer)]
use anyhow::*;
use enterpolation::{linear::ConstEquidistantLinear, Curve};
use minifb::{Key, Window, WindowOptions};
use palette::{IntoColor, LinSrgb, Srgb};
use serialport::SerialPort;
use std::io::{BufWriter, IoSliceMut, Read, Seek, SeekFrom, Write};
use std::result::Result::Ok;
use std::thread::sleep;
use std::time::Duration;
use std::{default, str};

use rand::Rng;
// use winit_input_helper::WinitInputHelper;

const FRAME_BEGIN_FLAG: u16 = 0xFF00;
const FRAME_END_FLAG: u8 = 0xDD;

const FRAME_HEAD_SIZE: usize = 20;

use rkyv::{Archive, Deserialize, Serialize};
use zerocopy::{FromBytes, FromZeroes};

#[derive(Archive, Serialize, Deserialize)]
#[repr(C)]
struct Data<'a> {
    label: u8,
    pixels: &'a [u8],
}

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

// fn draw_circle(
//     buffer: &mut [u32],
//     width: usize,
//     height: usize,
//     cx: usize,
//     cy: usize,
//     radius: usize,
//     color: u32,
// ) {
//     for y in 0..height {
//         for x in 0..width {
//             let dx = x as isize - cx as isize;
//             let dy = y as isize - cy as isize;
//             if dx * dx + dy * dy <= (radius * radius) as isize {
//                 buffer[y * width + x] = color;
//             }
//         }
//     }
// }
fn main() -> Result<()> {
    let file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open("training_data.bin")
        .unwrap();

    let mut bufwriter = BufWriter::with_capacity(20480, file);

    let target = serialport::available_ports()
        .expect("Unable to find any available ports")
        .into_iter()
        .find(|c| {
            if let serialport::SerialPortType::UsbPort(ref upi) = c.port_type
                && let Some(ref s) = upi.product
            {
                s.to_lowercase() == "SIPEED Meta Sense Lite".to_lowercase()
            } else {
                false
            }
        })
        .expect("Unable to find SIPEED Meta Sense Lite");

    let mut port = serialport::new(target.port_name, 115_200)
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
    // port.write(b"AT+BAUD=3000000\r").ok();
    // sleep(Duration::from_millis(20));

    const WIDTH: usize = 100;
    const HEIGHT: usize = 100;
    let mut window = Window::new(
        "Depth Image",
        WIDTH,
        HEIGHT,
        WindowOptions {
            scale: minifb::Scale::X4,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    let mut buf = [0u8; 10022];
    let mut output = [0u32; HEIGHT * WIDTH];
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
    // let mut rng = rand::thread_rng();
    // const RG: std::ops::Range<usize> = 3..97;
    // let mut cp = (rng.gen_range(RG), rng.gen_range(RG));
    // let mut count = 1;
    // let mut count1 = 0;
    window.set_target_fps(60);
    while window.is_open() && !window.is_key_down(Key::Escape) {
        port.read_exact(&mut buf).ok();
        if let Some(header) = FrameHead::ref_from_prefix(&buf)
            && header.frame_begin_flag == 0xff00
            && buf.ends_with(&[0xDD])
        {
            // println!("Frame id : {}", header.frame_id);
            if let Some(key) = window.get_keys_pressed(minifb::KeyRepeat::No).first() {
                if (*key as u8) <= 35 {
                    if window.is_key_down(Key::LeftAlt) {
                        if (*key as u8) < 10 {
                            port.write(format!("AT+UNIT={}\r", *key as u8 + 1).as_bytes())
                                .ok();
                        }
                    } else {
                        println!("Saved label {:#?}", key);
                        bufwriter.write(&[*key as u8]).ok();
                        bufwriter.write(&buf[20..(20 + (HEIGHT * WIDTH))]).ok();
                    }
                } else if *key == Key::Backspace {
                    // dbg!(bufwriter.buffer());
                    if bufwriter
                        .seek_relative(-((1 + WIDTH * HEIGHT) as i64))
                        .ok()
                        .is_some()
                    {
                        println!("Deleted last label");
                    } else {
                        println!("Unable to delete last label");
                    }
                    // dbg!(bufwriter.buffer());
                }
            }
            for (i, pixel) in buf[20..].iter().take(WIDTH * HEIGHT).enumerate() {
                let intensity = *pixel as usize;
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
            // }); // window.draw_circle(cp.0 as i32, cp.1 as i32, 10, 0xFF0000);
            window.update_with_buffer(&output, WIDTH, HEIGHT).unwrap();
            // dbg!(&buf[10015]);
            // dbg!(header);
        }
        sleep(Duration::from_millis(16));
    }

    bufwriter.flush().ok();

    Ok(())
}
