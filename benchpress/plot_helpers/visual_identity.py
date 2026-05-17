"""
BenchPress — Visual Identity & Plot Style

Single source of truth for ALL visual elements: colors, semantic mappings,
markers, line styles, font presets, LaTeX table/box styles, and plot helpers.

Usage:
    from benchpress.plot_helpers.visual_identity import *
    # or via backward-compat wrapper:
    from benchpress.plot_helpers.style import PROVIDER_COLORS, save_fig, apply_single
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ══════════════════════════════════════════════════════════════
#  PROJECT ROOT
# ══════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ══════════════════════════════════════════════════════════════
#  COLOR PALETTE
# ══════════════════════════════════════════════════════════════
# Memento / Solarized-inspired palette. The theme is intentionally centered on
# the four colors that define Memento's visual identity: magenta, vanilla blue,
# violet, and cyan teal. The legacy variable names are kept so existing plot
# scripts inherit the new theme without import churn.
SOL_BASE03 = '#002B36'
SOL_BASE01 = '#586E75'
SOL_BASE1  = '#93A1A1'
SOL_BASE2  = '#EEE8D5'
SOL_BASE3  = '#FDF6E3'
SOL_COOL_GRAY = '#EEF1F4'
SOL_BLUE_BG = '#E7F4FF'
SOL_GREEN_BG = '#F3F8E8'
SOL_BLUE   = '#268BD2'
SOL_BLUE_DARK = '#1B6EA8'
SOL_SKY    = '#4FA3D9'
SOL_CYAN   = '#2AA198'
SOL_AQUA   = '#7BC8C0'
SOL_MAGENTA = '#D33682'
SOL_ROSE   = '#F06AA6'
SOL_VIOLET = '#6C71C4'
SOL_LAVENDER = '#9B8FD9'

MEMENTO_MAGENTA = SOL_MAGENTA
VANILLA_BLUE = SOL_BLUE
ANSWER_VIOLET = SOL_VIOLET
CYAN_TEAL = SOL_CYAN
ROSE = SOL_ROSE
SKY_BLUE = SOL_SKY
DEEP_BLUE = SOL_BLUE_DARK
AQUA = SOL_AQUA
LAVENDER = SOL_LAVENDER
PANEL_BLUE_BG = SOL_BLUE_BG
USER_GREEN_BG = SOL_GREEN_BG
MISSING_CELL_GRAY = SOL_COOL_GRAY

CHARCOAL   = SOL_BASE03   # Text, axes, dark primary
TEAL       = CYAN_TEAL    # Theme cyan teal
VERDIGRIS  = VANILLA_BLUE # Legacy alias; now theme blue
OLIVE      = CYAN_TEAL    # Legacy alias; now theme teal
JASMINE    = LAVENDER     # Legacy alias; now theme violet tint
SANDY      = ROSE         # Legacy alias; now theme magenta tint
TANGERINE  = MEMENTO_MAGENTA  # Legacy alias; now theme magenta
PEACH      = MEMENTO_MAGENTA  # Legacy alias; now theme magenta
GRAY       = SOL_BASE1    # Muted / disabled

# Convenience aliases
PRIMARY    = TANGERINE
SECONDARY  = VERDIGRIS
ACCENT     = ANSWER_VIOLET
ALERT      = PEACH
WARM       = SANDY

# Ordered palette for cycling through N series. Keep the first four as the
# default theme colors; no off-theme warm/green fallback colors.
PALETTE = [
    MEMENTO_MAGENTA,
    VANILLA_BLUE,
    ANSWER_VIOLET,
    CYAN_TEAL,
    ROSE,
    SKY_BLUE,
    LAVENDER,
    AQUA,
    DEEP_BLUE,
    SOL_BASE01,
    CHARCOAL,
]

# ══════════════════════════════════════════════════════════════
#  SEMANTIC COLOR MAP  (hex → what it represents)
# ══════════════════════════════════════════════════════════════
COLORS = {
    MEMENTO_MAGENTA: {'method': 'BenchPress / cost-unaware greedy', 'provider': 'Meta, Moonshot AI',          'semantic': 'Memento magenta / cost-unaware'},
    VANILLA_BLUE:    {'method': 'Cost-aware greedy / baseline', 'provider': 'OpenAI',                    'semantic': 'Vanilla blue / cost-aware'},
    ANSWER_VIOLET:   {'method': 'BenchReg',                    'provider': 'Anthropic, Cohere',          'semantic': 'violet / model output'},
    CYAN_TEAL:       {'method': None,                          'provider': 'Google, Microsoft, Samsung', 'semantic': 'cyan teal / neutral accent'},
    ROSE:            {'method': '+RL / secondary Memento',     'provider': None,                         'semantic': 'light Memento accent'},
    SKY_BLUE:        {'method': None,                          'provider': None,                         'semantic': 'secondary blue'},
    LAVENDER:   {'method': None,                          'provider': None,                         'semantic': 'light violet accent'},
    AQUA:       {'method': None,                          'provider': None,                         'semantic': 'light cyan accent'},
    DEEP_BLUE:  {'method': None,                          'provider': None,                         'semantic': 'deep blue accent'},
    GRAY:       {'method': 'Mean',                        'provider': 'unknown (fallback)',         'semantic': 'muted'},
    CHARCOAL:   {'method': None,                          'provider': None,                         'semantic': 'text / axes'},
}

# Probe-selection and ranking-preservation policy styles. Use these instead of
# choosing palette colors directly in probe-set figures.
PROBE_RANDOM_STYLE = {
    'label': 'Random baseline',
    'color': GRAY,
    'linestyle': '--',
    'marker': 'o',
}
PROBE_COST_UNAWARE_STYLE = {
    'label': 'Cost-unaware greedy',
    'color': MEMENTO_MAGENTA,
    'linestyle': '-',
    'marker': 'o',
}
PROBE_COST_AWARE_STYLE = {
    'label': 'Cost-aware greedy',
    'color': VANILLA_BLUE,
    'linestyle': '-',
    'marker': 'o',
}
PROBE_CHEAP_CANDIDATE_STYLE = PROBE_COST_AWARE_STYLE
PROBE_POLICY_STYLES = {
    'random': PROBE_RANDOM_STYLE,
    'cost_unaware': PROBE_COST_UNAWARE_STYLE,
    'cost_aware': PROBE_COST_AWARE_STYLE,
    'cheap_candidate': PROBE_COST_AWARE_STYLE,
}

# ══════════════════════════════════════════════════════════════
#  PROVIDER / CATEGORY COLORS
# ══════════════════════════════════════════════════════════════
PROVIDER_COLORS = {
    'OpenAI':      VANILLA_BLUE,
    'Anthropic':   ANSWER_VIOLET,
    'Google':      CYAN_TEAL,
    'DeepSeek':    ROSE,
    'Alibaba':     SKY_BLUE,
    'Meta':        MEMENTO_MAGENTA,
    'xAI':         LAVENDER,
    'Mistral':     AQUA,
    'Microsoft':   CYAN_TEAL,
    'Moonshot AI': MEMENTO_MAGENTA,
    'ByteDance':   AQUA,
    'Amazon':      LAVENDER,
    'Cohere':      DEEP_BLUE,
    'NVIDIA':      AQUA,
    'Samsung':     CYAN_TEAL,
}

CAT_COLORS = {
    'Math':                   ANSWER_VIOLET,
    'Coding':                 CYAN_TEAL,
    'Reasoning':              MEMENTO_MAGENTA,
    'Knowledge':              VANILLA_BLUE,
    'Agentic':                ROSE,
    'Multimodal':             SKY_BLUE,
    'Instruction Following':  LAVENDER,
    'Science':                ANSWER_VIOLET,
    'Long Context':           SOL_BASE01,
    'Composite':              AQUA,
    'Human Preference':       GRAY,
}

# ══════════════════════════════════════════════════════════════
#  MARKERS & LINE STYLES
# ══════════════════════════════════════════════════════════════
MARKERS = {
    '^': {'model_type': 'reasoning',     'size': 40},
    'o': {'model_type': 'non-reasoning', 'size': 25},
    's': {'semantic': 'within ±3 coverage'},
    'D': {'semantic': 'within ±5 coverage'},
}

LINES = {
    '-':  {'use': 'primary curve'},
    '--': {'use': 'reference / baseline / axis'},
    'o-': {'use': 'data series with markers'},
}

# ══════════════════════════════════════════════════════════════
#  FONT PRESETS
# ══════════════════════════════════════════════════════════════
SINGLE_COL = {
    'font.size':          22,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.labelsize':     24,
    'axes.titlesize':     26,
    'xtick.labelsize':    20,
    'ytick.labelsize':    20,
    'legend.fontsize':    18,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
    'lines.linewidth':    3.0,
    'lines.markersize':   10,
}

DOUBLE_COL = {
    'font.size':          16,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.labelsize':     18,
    'axes.titlesize':     20,
    'xtick.labelsize':    15,
    'ytick.labelsize':    15,
    'legend.fontsize':    14,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
    'lines.linewidth':    2.5,
    'lines.markersize':   8,
}

TALL_FIG = {
    **DOUBLE_COL,
    'ytick.labelsize':    13,
    'axes.labelsize':     17,
}

# Deprecated: kept for backward compat with old scripts
FIG_DIR = os.path.join(PROJECT_ROOT, 'figures')

# ══════════════════════════════════════════════════════════════
#  LaTeX STYLES  (table cells & highlight boxes)
# ══════════════════════════════════════════════════════════════
TABLE = {
    'ours':     r'\cellcolor{bpMagenta!5}',
    'baseline': r'\cellcolor{black!3}',
    'header':   r'\cellcolor{gray!10}',
}

BOXES = {
    'finding':    {'border': 'bpMagenta', 'bg': 'bpMagenta!5'},
    'key_result': {'border': 'bpCyan',    'bg': 'bpCyan!5'},
    'caveat':     {'border': 'bpViolet',  'bg': 'bpViolet!5'},
}

# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
#  LLM MODEL STYLING (for LLM-as-predictor experiments)
# ══════════════════════════════════════════════════════════════
# Format: model_id → (display_name, color, marker, linestyle)
LLM_MODEL_STYLE = {
    'bp':                   ('BenchPress',       MEMENTO_MAGENTA, 's', '-'),
    'gpt-5.5':              ('GPT-5.5',          VANILLA_BLUE,    'o', '-'),
}

# Convenience dicts derived from LLM_MODEL_STYLE
LLM_MODEL_NAMES  = {k: v[0] for k, v in LLM_MODEL_STYLE.items()}
LLM_MODEL_COLORS = {k: v[1] for k, v in LLM_MODEL_STYLE.items()}

# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def apply_single():
    """Apply single-column style (most figures)."""
    plt.rcParams.update(SINGLE_COL)

def apply_double():
    """Apply double-column style (figure* figures)."""
    plt.rcParams.update(DOUBLE_COL)

def apply_tall():
    """Apply tall-figure style (many bars)."""
    plt.rcParams.update(TALL_FIG)

def save_fig(name):
    """Save figure as PDF + PNG into the caller's figures/ subdirectory, then close."""
    import inspect
    caller_file = inspect.stack()[1].filename
    caller_dir = os.path.dirname(os.path.abspath(caller_file))
    fig_dir = os.path.join(caller_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, f'{name}.png'), bbox_inches='tight')
    plt.close()
    print(f"  -> figures/{name}.pdf + .png")

def integer_ticks(ax, axis='both'):
    """Force integer-only ticks on the given axis."""
    from matplotlib.ticker import MaxNLocator
    if axis in ('x', 'both'):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if axis in ('y', 'both'):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

def provider_legend(ax, providers_used, **kwargs):
    """Add a provider-colored legend."""
    handles = [mpatches.Patch(color=PROVIDER_COLORS.get(p, GRAY), label=p)
               for p in sorted(PROVIDER_COLORS) if p in providers_used]
    defaults = dict(loc='lower right', fontsize=kwargs.pop('fontsize', None), ncol=2)
    defaults.update(kwargs)
    if defaults['fontsize'] is None:
        del defaults['fontsize']
    ax.legend(handles=handles, **defaults)
