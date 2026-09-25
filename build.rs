fn main() {
    let css  = std::fs::read_to_string("ui/style.css").unwrap_or_default();
    let body = std::fs::read_to_string("ui/body.html").unwrap_or_default();
    let js   = std::fs::read_to_string("ui/app.js").unwrap_or_default();
    let html = format!(
        "<!DOCTYPE html><html lang=\"de\"><head><meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>BrainCut Blur</title><style>{css}</style></head><body>{body}<script>{js}</script></body></html>"
    );
    std::fs::write("ui/index.html", html).unwrap();
    println!("cargo:rerun-if-changed=ui/style.css");
    println!("cargo:rerun-if-changed=ui/body.html");
    println!("cargo:rerun-if-changed=ui/app.js");
}
