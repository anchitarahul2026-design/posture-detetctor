"""Standalone profile selector for AlignMe.

Run this file directly to display the four-option selection layout:
    python patient_profile_selector.py

Import ``show_profile_selector`` from another module when wiring it into an
application.  It returns the selected profile name, or ``None`` if closed.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional


PROFILES = (
    ("Normal person", "Everyday posture and movement monitoring.", "#238e5a", "◎"),
    ("Surgery", "Support a careful recovery and check-in routine.", "#d27633", "✚"),
    ("Injury", "Track mobility while you work through an injury.", "#3679c7", "⌁"),
    ("Fractured", "A gentler setup for fracture-related recovery.", "#7651bb", "╱"),
)


class ProfileSelector(tk.Tk):
    """A self-contained, keyboard-friendly profile selection window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("AlignMe | Select profile")
        self.configure(background="#f7f9fc")
        self.minsize(680, 560)
        self.geometry("760x620")
        self.selected_profile: Optional[str] = None
        self._cards: dict[str, tk.Frame] = {}
        self._card_labels: dict[str, list[tk.Widget]] = {}
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        for index in range(4):
            self.bind(str(index + 1), lambda _event, i=index: self.select(PROFILES[i][0]))

    def _build_layout(self) -> None:
        header = tk.Frame(self, background="#ffffff", height=68)
        header.pack(fill="x")
        header.pack_propagate(False)

        brand = tk.Label(
            header, text="✚  ALIGNME", background="#ffffff", foreground="#172033",
            font=("Arial", 14, "bold"), padx=30,
        )
        brand.pack(side="left", pady=20)
        tk.Label(
            header, text="●  Secure monitoring setup", background="#ffffff",
            foreground="#68758a", font=("Arial", 10), padx=30,
        ).pack(side="right", pady=22)

        content = tk.Frame(self, background="#f7f9fc", padx=58, pady=46)
        content.pack(fill="both", expand=True)
        tk.Label(
            content, text="STEP 1 OF 2", background="#f7f9fc", foreground="#237ff0",
            font=("Arial", 9, "bold"), anchor="w",
        ).pack(fill="x")
        tk.Label(
            content, text="Choose your monitoring profile", background="#f7f9fc",
            foreground="#172033", font=("Arial", 24, "bold"), anchor="w", pady=8,
        ).pack(fill="x")
        tk.Label(
            content, text="This helps tailor the session setup to your current needs.",
            background="#f7f9fc", foreground="#65738a", font=("Arial", 11), anchor="w",
        ).pack(fill="x", pady=(0, 24))

        grid = tk.Frame(content, background="#f7f9fc")
        grid.pack(fill="both", expand=True)
        for column in range(2):
            grid.grid_columnconfigure(column, weight=1, uniform="profiles")
        for row in range(2):
            grid.grid_rowconfigure(row, weight=1)

        for index, (name, description, color, icon) in enumerate(PROFILES):
            self._add_card(grid, index, name, description, color, icon)

        footer = tk.Frame(content, background="#f7f9fc", pady=22)
        footer.pack(fill="x")
        self.status = tk.Label(
            footer, text="Select a profile to continue", background="#f7f9fc",
            foreground="#65738a", font=("Arial", 10), anchor="w",
        )
        self.status.pack(side="left")
        self.continue_button = ttk.Button(
            footer, text="Continue to setup  →", command=self._continue, state="disabled",
        )
        self.continue_button.pack(side="right")

    def _add_card(
        self, parent: tk.Frame, index: int, name: str, description: str, color: str, icon: str
    ) -> None:
        card = tk.Frame(parent, background="#ffffff", highlightbackground="#d8e0eb", highlightthickness=1)
        card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=8, pady=8)
        self._cards[name] = card
        labels: list[tk.Widget] = []
        self._card_labels[name] = labels

        top = tk.Frame(card, background="#ffffff", padx=18, pady=17)
        top.pack(fill="x")
        icon_label = tk.Label(top, text=icon, background="#ffffff", foreground=color, font=("Arial", 20, "bold"))
        icon_label.pack(side="left", padx=(0, 12))
        title = tk.Label(top, text=name, background="#ffffff", foreground="#172033", font=("Arial", 13, "bold"))
        title.pack(side="left")
        labels.extend((top, icon_label, title))

        detail = tk.Label(card, text=description, background="#ffffff", foreground="#65738a", font=("Arial", 10), anchor="w", justify="left", padx=18)
        detail.pack(fill="x")
        hint = tk.Label(card, text=f"Select profile  •  Press {index + 1}", background="#ffffff", foreground=color, font=("Arial", 9, "bold"), anchor="w", padx=18, pady=16)
        hint.pack(fill="x")
        labels.extend((detail, hint))

        for widget in (card, *labels):
            widget.bind("<Button-1>", lambda _event, profile=name: self.select(profile))
            widget.bind("<Enter>", lambda _event, profile=name: self._hover(profile, True))
            widget.bind("<Leave>", lambda _event, profile=name: self._hover(profile, False))

    def _hover(self, profile: str, active: bool) -> None:
        if profile != self.selected_profile:
            self._cards[profile].configure(highlightbackground="#9fb1c9" if active else "#d8e0eb")

    def select(self, profile: str) -> None:
        self.selected_profile = profile
        for name, card in self._cards.items():
            card.configure(highlightbackground="#237ff0" if name == profile else "#d8e0eb", highlightthickness=2 if name == profile else 1)
        self.status.configure(text=f"{profile} selected")
        self.continue_button.configure(state="normal")

    def _continue(self) -> None:
        self.destroy()

    def _close(self) -> None:
        self.selected_profile = None
        self.destroy()


def show_profile_selector() -> Optional[str]:
    """Open the layout and return the chosen profile, if any."""
    selector = ProfileSelector()
    selector.mainloop()
    return selector.selected_profile


if __name__ == "__main__":
    selection = show_profile_selector()
    if selection:
        print(f"Selected profile: {selection}")
