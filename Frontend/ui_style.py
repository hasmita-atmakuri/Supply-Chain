from tkinter import Text, Frame, Label, FLAT, GROOVE

def style_main_window(main):
    """
    Apply a modern, minimal, and professional styling to the main Tk window.
    Returns the configured text widget, shared button options, and button parent frame.
    """

    # Window configuration
    main.title("Supply Chain Disruption Analysis")
    main.geometry("1200x800")
    main.configure(bg="#e6f2ff")  # soft light blue background

    # Content frame (acts like a responsive container)
    content = Frame(main, bg="#e6f2ff")
    content.pack(fill="both", expand=True, padx=24, pady=24)

    # Left pane for heading + output
    left_pane = Frame(content, bg="#e6f2ff")
    left_pane.pack(side="left", fill="both", expand=True)

    # Heading with clear hierarchy
    label_heading = Label(
        left_pane,
        text="Supply Chain Disruption Analysis",
        font=("Segoe UI", 18, "bold"),
        bg="#e6f2ff",
        fg="#1f2937",  # dark gray for readability
        pady=12
    )
    label_heading.pack(anchor="w")

    # Text console (output window)
    text = Text(
        left_pane,
        height=25,
        width=100,
        bg="#ffffff",  # clean white background
        fg="#1f2937",  # dark gray text
        insertbackground="#1f2937",  # cursor color
        font=("Consolas", 11),
        relief=FLAT,
        borderwidth=0,
        highlightthickness=1,
        highlightbackground="#d1d5db",  # subtle border
        padx=12,
        pady=12,
    )
    text.pack(fill="both", expand=True, pady=(8, 0))

    # Reusable button style (modern, rounded, subtle hover)
    button_options = {
        "bg": "#3b82f6",              # muted blue
        "fg": "#ffffff",
        "activebackground": "#2563eb",  # darker blue on hover
        "activeforeground": "#ffffff",
        "font": ("Segoe UI", 11, "bold"),
        "relief": GROOVE,
        "bd": 0,
        "width": 26,
        "highlightthickness": 0,
        "padx": 8,
        "pady": 8,
    }

    # Right pane for buttons (card-style container)
    buttons_parent = Frame(
        content,
        bg="#ffffff",
        relief=FLAT,
        bd=0,
        highlightthickness=1,
        highlightbackground="#e5e7eb",  # subtle border
        padx=16,
        pady=16,
    )
    buttons_parent.pack(side="right", fill="y", padx=(20, 0))

    return text, button_options, buttons_parent
