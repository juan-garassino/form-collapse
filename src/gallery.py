"""HTML gallery generator from session data."""
import os
import json
import logging
from typing import List

logger = logging.getLogger(__name__)

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>formCollapse Gallery</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 2rem; }
  h1 { text-align: center; margin-bottom: 1rem; color: #00d4ff; font-size: 2rem; }
  .filters { text-align: center; margin-bottom: 2rem; }
  .filters button { background: #16213e; border: 1px solid #0f3460; color: #e0e0e0; padding: 0.5rem 1rem; margin: 0.25rem; cursor: pointer; border-radius: 4px; }
  .filters button.active, .filters button:hover { background: #0f3460; border-color: #00d4ff; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.5rem; }
  .card { background: #16213e; border-radius: 8px; overflow: hidden; transition: transform 0.2s; border: 1px solid #0f3460; }
  .card:hover { transform: translateY(-4px); border-color: #00d4ff; }
  .card img { width: 100%; height: 200px; object-fit: contain; background: #0a0a1a; }
  .card .no-img { width: 100%; height: 200px; display: flex; align-items: center; justify-content: center; background: #0a0a1a; color: #555; }
  .card-body { padding: 1rem; }
  .card-title { font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem; }
  .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: bold; margin-right: 0.5rem; }
  .badge.strange_attractor { background: #00c853; color: #000; }
  .badge.limit_cycle { background: #2196f3; color: #fff; }
  .badge.fixed_point { background: #9e9e9e; color: #000; }
  .badge.divergent { background: #f44336; color: #fff; }
  .badge.failed { background: #795548; color: #fff; }
  .mle { font-size: 0.85rem; color: #aaa; margin-top: 0.5rem; }
  .params { font-size: 0.75rem; color: #777; margin-top: 0.5rem; font-family: monospace; }
  .hidden { display: none; }
</style>
</head>
<body>
<h1>formCollapse Gallery</h1>
<div class="filters">
  <button class="active" onclick="filterCards('all')">All</button>
  <button onclick="filterCards('strange_attractor')">Strange Attractor</button>
  <button onclick="filterCards('limit_cycle')">Limit Cycle</button>
  <button onclick="filterCards('fixed_point')">Fixed Point</button>
  <button onclick="filterCards('divergent')">Divergent</button>
</div>
<div class="grid" id="gallery">
  {{CARDS}}
</div>
<script>
function filterCards(type) {
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.card').forEach(card => {
    if (type === 'all' || card.dataset.type === type) {
      card.classList.remove('hidden');
    } else {
      card.classList.add('hidden');
    }
  });
}
</script>
</body>
</html>'''

CARD_TEMPLATE = '''<div class="card" data-type="{classification}">
  {img_tag}
  <div class="card-body">
    <div class="card-title">{system_name}</div>
    <span class="badge {classification}">{classification}</span>
    <div class="mle">MLE: {lyapunov}</div>
    <div class="params">{params_str}</div>
  </div>
</div>'''


def generate_gallery(base_dir: str = "results", output_dir: str = None) -> str:
    """Scan all session.json files and generate an HTML gallery."""
    if output_dir is None:
        output_dir = os.path.join(base_dir, "gallery")
    os.makedirs(output_dir, exist_ok=True)

    sessions = _find_sessions(base_dir)
    logger.info(f"Found {len(sessions)} sessions")

    cards_html = []
    for session_data, session_dir in sessions:
        for result in session_data.get('results', []):
            system_name = result.get('system_name', 'Unknown')
            classification = result.get('classification', 'unknown')
            lyap = result.get('lyapunov_exponent')
            lyap_str = f"{lyap:.6f}" if lyap is not None else "N/A"
            params = result.get('params', {})
            params_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in params.items()
                                   if k not in ('sim_time', 'sim_steps', 'scale'))

            # Find thumbnail
            img_tag = '<div class="no-img">No preview</div>'
            files = result.get('files', [])
            for f in files:
                if 'attractor_3d' in f and f.endswith('.png'):
                    img_path = os.path.join(session_dir, 'png', f)
                    if os.path.exists(img_path):
                        rel_path = os.path.relpath(img_path, output_dir)
                        img_tag = f'<img src="{rel_path}" alt="{system_name}">'
                    break

            cards_html.append(CARD_TEMPLATE.format(
                classification=classification,
                system_name=system_name,
                lyapunov=lyap_str,
                params_str=params_str,
                img_tag=img_tag,
            ))

    html = HTML_TEMPLATE.replace('{{CARDS}}', '\n  '.join(cards_html))

    output_path = os.path.join(output_dir, "index.html")
    with open(output_path, 'w') as f:
        f.write(html)

    logger.info(f"Gallery generated: {output_path} ({len(cards_html)} entries)")
    return output_path


def _find_sessions(base_dir: str) -> List[tuple]:
    """Find all session.json files in the results directory."""
    sessions = []
    if not os.path.exists(base_dir):
        return sessions

    for entry in os.listdir(base_dir):
        session_dir = os.path.join(base_dir, entry)
        session_file = os.path.join(session_dir, 'session.json')
        if os.path.isdir(session_dir) and os.path.exists(session_file):
            try:
                with open(session_file) as f:
                    data = json.load(f)
                sessions.append((data, session_dir))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read {session_file}: {e}")

    return sessions
