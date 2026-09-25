const report = window.dispatchReport;
const arms = [
  { id: 'base_zero', label: 'Base · zero-shot' },
  { id: 'base_few', label: 'Base · quatro exemplos' },
  { id: 'lora_base', label: 'LoRA · sem aumento' },
  { id: 'lora_augmented', label: 'LoRA · com aumento' }
];
const fields = [
  ['queue', 'Fila'],
  ['priority', 'Prioridade'],
  ['order_id', 'Pedido'],
  ['needs_review', 'Revisão humana']
];
const state = { split: 'challenge', filter: 'all', search: '', selected: null };
const $ = id => document.getElementById(id);
const node = (tag, className, value) => {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (value !== undefined) item.textContent = value;
  return item;
};
const percent = value => Math.round(value * 100) + '%';
const exact = (value, total) => Math.round(value * total) + '/' + total;
const predicted = (item, arm) => item.predictions[arm].value;
const correct = (item, arm) => { const value = predicted(item, arm); return value !== null && fields.every(([key]) => value[key] === item.target[key]); };
function status(item) {
  const base = correct(item, 'lora_base');
  const augmented = correct(item, 'lora_augmented');
  if (base && augmented) return ['both_right', 'Ambos corretos'];
  if (!base && augmented) return ['fixed', 'Aumento corrigiu'];
  if (base && !augmented) return ['regressed', 'Aumento piorou'];
  return ['both_wrong', 'Ambos errados'];
}
function valueText(value) {
  if (value === null || value === undefined) return 'ausente';
  if (typeof value === 'boolean') return value ? 'sim' : 'não';
  return String(value);
}
function renderMetrics() {
  const section = report.comparison[state.split], total = state.split === 'challenge' ? 20 : 54;
  const container = $('metric-grid');
  container.replaceChildren();
  arms.forEach((arm, index) => {
    const run = section.runs[arm.id], card = node('article', 'metric' + (arm.id === 'lora_base' ? ' featured' : ''));
    card.append(node('span', 'arm-index', '0' + (index + 1) + ' / ' + (arm.id.includes('lora') ? 'ADAPTADO' : 'BASE')), node('h3', '', arm.label));
    const score = node('span', 'metric-value', String(Math.round(run.exact_match * total)));
    score.append(node('small', '', ' / ' + total));
    card.append(score, node('div', 'metric-detail', 'JSON no esquema ' + percent(run.schema_validity) + ' · revisão ' + percent(run.review_recall)));
    container.append(card);
  });
  $('split-description').textContent = state.split === 'challenge'
    ? 'Vinte mensagens escritas fora dos templates de treino. O filtro abaixo permite conferir cada ganho e regressão do aumento por IA.'
    : 'Cinquenta e quatro casos de teste com padrões do mesmo gerador de frases do treino. Acerto aqui não implica a mesma cobertura em linguagem nova.';
  const paired = section.augmented_vs_unaugmented;
  $('reading-note').textContent = state.split === 'challenge'
    ? 'Comparação pareada: o aumento corrigiu ' + paired.right_only_correct + ' casos e perdeu ' + paired.left_only_correct + ' que o LoRA sem aumento acertava. Este único conjunto não estabelece a causa da diferença.'
    : 'Os dois LoRAs acertaram todos os casos deste conjunto. O desafio fora dos templates separa melhor os braços nesta execução.';
  for (const button of $('split-tabs').querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.split === state.split));
  $('case-split').value = state.split;
}
function renderBars() {
  const section = report.comparison[state.split], host = $('accuracy-bars');
  host.replaceChildren();
  for (const arm of arms) {
    const score = section.runs[arm.id].exact_match;
    const row = node('div', 'bar-row' + (arm.id === 'lora_base' ? ' featured' : ''));
    const track = node('div', 'bar-track'), fill = node('div', 'bar-fill');
    fill.style.width = percent(score);
    track.append(fill);
    row.append(node('span', 'bar-label', arm.label), track, node('span', 'bar-value', percent(score)));
    host.append(row);
  }
}
function renderFields() {
  const section = report.comparison[state.split], host = $('field-table'), table = node('table');
  const head = node('thead'), header = node('tr');
  header.append(node('th', '', 'Campo'));
  arms.forEach(arm => header.append(node('th', '', arm.label.replace('Base · ', 'B ').replace('LoRA · ', 'L '))));
  head.append(header); table.append(head);
  const body = node('tbody');
  for (const [key, label] of fields) {
    const row = node('tr'), best = Math.max(...arms.map(arm => section.runs[arm.id].field_accuracy[key]));
    row.append(node('td', '', label));
    for (const arm of arms) {
      const value = section.runs[arm.id].field_accuracy[key], cell = node('td', value === best ? 'best' : '', percent(value));
      row.append(cell);
    }
    body.append(row);
  }
  table.append(body); host.replaceChildren(table);
}
function renderMethod() {
  const augmentation = report.augmentation, recipe = report.training.recipe, trained = report.training.runs['lora-base'];
  $('augmentation-method').textContent = augmentation.attempted + ' paráfrases de treino tentadas com IA; ' + augmentation.accepted + ' aceitas e ' + augmentation.rejected + ' rejeitadas por checagens de fatos e referência de pedido. Validação e teste ficaram fora da geração.';
  $('training-method').textContent = 'Qwen3.5-0.8B em bf16, LoRA rank ' + recipe.lora_rank + ', ' + recipe.max_steps + ' passos e seed ' + recipe.seed + '. Os dois braços mudam apenas nos exemplos aumentados.';
  const items = [
    'Modelo ' + report.model,
    'GPU ' + trained.gpu.replace('NVIDIA GeForce ', ''),
    'Pico PyTorch ' + (trained.peak_vram_bytes / 1024 ** 3).toFixed(2) + ' GiB',
    'Avaliação 54 + 20 casos'
  ];
  $('protocol-strip').replaceChildren(...items.map(item => node('span', '', item)));
}
function renderCases() {
  const query = state.search.toLocaleLowerCase('pt-BR');
  const visible = report.cases.filter(item => item.split === state.split)
    .filter(item => state.filter === 'all' || status(item)[0] === state.filter)
    .filter(item => !query || item.id.toLocaleLowerCase('pt-BR').includes(query) || item.text.toLocaleLowerCase('pt-BR').includes(query));
  $('case-count').textContent = visible.length + ' caso(s)';
  if (!visible.some(item => item.id === state.selected)) state.selected = visible[0]?.id || null;
  const host = $('case-list'); host.replaceChildren();
  if (!visible.length) host.append(node('div', 'empty', 'Nenhum caso corresponde aos filtros.'));
  for (const item of visible) {
    const button = node('button', 'case-item' + (state.selected === item.id ? ' active' : ''));
    button.type = 'button';
    button.setAttribute('aria-pressed', String(state.selected === item.id));
    const head = node('div', 'case-item-head');
    head.append(node('span', '', item.id), node('span', '', status(item)[1]));
    button.append(head, node('p', '', item.text), node('span', 'case-status', correct(item, 'lora_base') ? 'LoRA base ✓' : 'LoRA base ×'));
    button.addEventListener('click', () => {
      state.selected = item.id;
      renderCases();
      if (window.innerWidth < 700) $('case-detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    host.append(button);
  }
  renderDetail(visible.find(item => item.id === state.selected));
}
function renderDetail(item) {
  const host = $('case-detail'); host.replaceChildren();
  if (!item) { host.append(node('div', 'empty', 'Selecione outro filtro para examinar um caso.')); return; }
  host.append(node('span', 'detail-id', item.id), node('h3', '', status(item)[1]), node('div', 'case-text', item.text));
  const target = node('div', 'target-row');
  target.append(node('strong', '', 'Rótulo esperado'));
  for (const [key, label] of fields) target.append(node('span', '', label + ': ' + valueText(item.target[key])));
  host.append(target);
  const grid = node('div', 'prediction-grid');
  for (const arm of arms) {
    const value = predicted(item, arm.id), card = node('div', 'prediction-card ' + (correct(item, arm.id) ? 'correct' : 'wrong'));
    const title = node('h4');
    title.append(node('span', '', arm.label), node('span', '', correct(item, arm.id) ? 'correto' : 'erro'));
    card.append(title);
    if (value === null) card.append(node('div', 'invalid-output', 'Saída sem os quatro campos válidos'));
    else {
      const list = node('dl');
      for (const [key, label] of fields) {
        list.append(node('dt', '', label), node('dd', value[key] === item.target[key] ? '' : 'mismatch', valueText(value[key])));
      }
      card.append(list);
    }
    grid.append(card);
  }
  host.append(grid, node('p', 'detail-note', 'Valores em terracota diferem do rótulo esperado. Uma saída inválida conta como erro.'));
}
function setSplit(split) {
  state.split = split;
  state.filter = 'all';
  state.search = '';
  $('case-filter').value = 'all';
  $('case-search').value = '';
  state.selected = null;
  renderMetrics(); renderBars(); renderFields(); renderCases();
}
$('split-tabs').addEventListener('click', event => {
  const button = event.target.closest('button[data-split]');
  if (button) setSplit(button.dataset.split);
});
$('case-split').addEventListener('change', event => setSplit(event.target.value));
$('case-filter').addEventListener('change', event => { state.filter = event.target.value; renderCases(); });
$('case-search').addEventListener('input', event => { state.search = event.target.value.trim(); renderCases(); });
renderMethod();
setSplit('challenge');
