// Builds the engine, build/libmyds4.a, with the repository's Makefile, and
// links it with the frameworks and libraries its C code uses.
use std::path::Path;
use std::process::Command;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest).parent().expect("server/ is inside the repository");
    let status = Command::new("make")
        .arg("-C")
        .arg(root)
        .arg("build/libmyds4.a")
        .status()
        .expect("make runs");
    assert!(status.success(), "make build/libmyds4.a failed");

    println!("cargo:rustc-link-search=native={}", root.join("build").display());
    println!("cargo:rustc-link-lib=static=myds4");
    for framework in ["Metal", "Foundation", "IOKit", "ImageIO", "CoreGraphics", "Accelerate"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    println!("cargo:rustc-link-lib=objc");
    // libjpeg-turbo, found through pkg-config as the Makefile finds it.
    let libs = Command::new("pkg-config")
        .args(["--libs", "libjpeg"])
        .output()
        .expect("pkg-config runs");
    assert!(libs.status.success(), "pkg-config finds no libjpeg (brew install jpeg-turbo)");
    for flag in String::from_utf8_lossy(&libs.stdout).split_whitespace() {
        if let Some(dir) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={dir}");
        } else if let Some(lib) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }

    // Cargo scans a directory's whole tree for changes.
    for path in ["Makefile", "src", "data/engram.bin", "data/unicode.bin"] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
}
