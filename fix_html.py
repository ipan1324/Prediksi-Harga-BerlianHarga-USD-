with open('templates/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

extra_css = """
/* ===== POLISH ===== */

.display-4 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
}

.hero-title {
    font-size: 3rem;
}

.gem-hero {
    animation: float 4s ease-in-out infinite;
}

@keyframes float {
    0%,100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-12px);
    }
}
"""

# Tambahkan CSS sebelum </style>
html = html.replace(
    "</style>",
    extra_css + "\n</style>",
    1
)

# Tambahkan animasi diamond
html = html.replace(
    'class="bi bi-gem text-primary"',
    'class="bi bi-gem text-primary gem-hero"'
)

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Done! HTML polished successfully.")