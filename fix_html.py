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

@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem;
    }
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

#resPrice {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 40%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}

.range-fill {
    animation: fillBar 1.2s cubic-bezier(0.4,0,0.2,1) forwards;
}

@keyframes fillBar {
    from {
        width: 0%;
    }

    to {
        width: 100%;
    }
}

.stat-card,
.stat-card-accent,
.stat-card-success {
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.stat-card:hover,
.stat-card-accent:hover,
.stat-card-success:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.4);
}

.spa-section.active {
    animation: fadeSection 0.35s ease forwards;
}

@keyframes fadeSection {
    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}
"""

# Tambahkan CSS sebelum </style>
html = html.replace(
    "</style>",
    extra_css + "\n</style>",
    1
)

# Tambahkan animasi icon diamond
html = html.replace(
    'class="bi bi-gem text-primary"',
    'class="bi bi-gem text-primary gem-hero"'
)

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Done! HTML polished successfully.")