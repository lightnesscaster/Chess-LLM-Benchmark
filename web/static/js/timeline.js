/* Timeline chart: every qualifying model at its release date, with the
 * record line (strongest model released so far) drawn over the top.
 * Plain SVG, built with presentation attributes so the same markup can be
 * serialized straight into the PNG export. */
(function () {
    'use strict';

    var dataEl = document.getElementById('timeline-data');
    var mount = document.getElementById('timeline-chart');
    if (!dataEl || !mount) return;

    var data = JSON.parse(dataEl.textContent);
    var tooltip = document.getElementById('timeline-tooltip');
    var legendButtons = Array.prototype.slice.call(document.querySelectorAll('[data-lab]'));

    var SVG_NS = 'http://www.w3.org/2000/svg';
    var FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
    var DAY = 86400;
    var COLOR = {
        surface: '#16213e',
        ink: '#eaeaea',
        ink2: '#b3bacb',
        muted: '#8a94ab',
        grid: '#223052',
        axis: '#3a4a70',
        record: '#f4f1e8',
        labs: { google: '#3987e5', openai: '#199e70', anthropic: '#d95926', other: '#8b93a7' }
    };
    var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    var measureCtx = document.createElement('canvas').getContext('2d');
    function textWidth(text, size, weight) {
        measureCtx.font = (weight || 400) + ' ' + size + 'px ' + FONT;
        return measureCtx.measureText(text).width;
    }

    function el(name, attrs, parent) {
        var node = document.createElementNS(SVG_NS, name);
        for (var key in attrs) node.setAttribute(key, attrs[key]);
        if (parent) parent.appendChild(node);
        return node;
    }

    function label(parent, text, x, y, opts) {
        var node = el('text', {
            x: x, y: y,
            fill: opts.fill || COLOR.muted,
            'font-size': opts.size || 12,
            'font-weight': opts.weight || 400,
            'font-family': FONT,
            'text-anchor': opts.anchor || 'start'
        }, parent);
        if (opts.tabular) node.setAttribute('style', 'font-variant-numeric: tabular-nums');
        node.textContent = text;
        return node;
    }

    function formatRating(value) {
        return String(Math.round(value)).replace('-', '−');
    }

    function formatDate(ts, withDay) {
        var d = new Date(ts * 1000);
        var month = MONTHS[d.getUTCMonth()];
        return withDay
            ? month + ' ' + d.getUTCDate() + ', ' + d.getUTCFullYear()
            : month + ' ' + d.getUTCFullYear();
    }

    /* Best-so-far staircase through a set of points, extended to today. */
    function staircase(points, x, y, endX) {
        var best = -Infinity, d = '', lastY = null;
        points.forEach(function (p) {
            if (p.rating <= best) return;
            best = p.rating;
            var px = x(p.ts), py = y(p.rating);
            d += lastY === null ? 'M' + px + ' ' + py : 'H' + px + 'V' + py;
            lastY = py;
        });
        return lastY === null ? '' : d + 'H' + endX;
    }

    function overlaps(a, b) {
        return a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
    }

    var state = { focusLab: null, previewLab: null, geometry: null };

    function build(width, forExport) {
        var compact = width < 640;
        var height = forExport ? 700 : Math.round(Math.max(360, Math.min(620, width * 0.54)));
        var header = forExport ? 96 : 0;
        var m = {
            top: header + 34,
            right: compact ? 14 : (forExport ? 112 : 96),
            bottom: 36,
            left: compact ? 40 : 50
        };
        var plot = { x0: m.left, x1: width - m.right, y0: m.top, y1: height - m.bottom };

        var points = data.points, records = data.records;
        var minTs = Math.min.apply(null, points.map(function (p) { return p.ts; }));
        var maxRating = Math.max.apply(null, points.map(function (p) { return p.rating; }));
        var minRating = Math.min.apply(null, points.map(function (p) { return p.rating; }));
        var tDomain = [minTs - 50 * DAY, data.now + 25 * DAY];
        var rDomain = [
            Math.floor((minRating - 100) / 100) * 100,
            Math.max(2400, Math.ceil((maxRating + 250) / 100) * 100)
        ];
        function x(ts) { return plot.x0 + (ts - tDomain[0]) / (tDomain[1] - tDomain[0]) * (plot.x1 - plot.x0); }
        function y(r) { return plot.y1 - (r - rDomain[0]) / (rDomain[1] - rDomain[0]) * (plot.y1 - plot.y0); }

        var svg = el('svg', {
            xmlns: SVG_NS,
            viewBox: '0 0 ' + width + ' ' + height,
            role: 'img',
            'aria-label': 'Scatter chart of LLM chess ratings by release date. ' +
                'The record progression is listed below the chart.'
        });
        // On the page the SVG is sized by CSS (a fixed width attribute would prop
        // the layout open on narrow screens); the export needs real dimensions.
        if (forExport) {
            svg.setAttribute('width', width);
            svg.setAttribute('height', height);
        }
        el('rect', { width: width, height: height, fill: COLOR.surface }, svg);

        /* Export header: title plus a drawn legend, since the HTML one is not in the image. */
        if (forExport) {
            label(svg, 'LLM chess ratings by release date', m.left, 38, { fill: COLOR.ink, size: 22, weight: 700 });
            var lx = m.left, ly = 70;
            el('line', { x1: lx, x2: lx + 22, y1: ly - 4, y2: ly - 4, stroke: COLOR.record, 'stroke-width': 2 }, svg);
            lx += 30;
            label(svg, 'Strongest so far', lx, ly, { fill: COLOR.ink2, size: 13 });
            lx += textWidth('Strongest so far', 13) + 22;
            data.labs.forEach(function (lab) {
                el('circle', { cx: lx + 5, cy: ly - 4, r: 5, fill: COLOR.labs[lab.key] }, svg);
                label(svg, lab.name, lx + 16, ly, { fill: COLOR.ink2, size: 13 });
                lx += 16 + textWidth(lab.name, 13) + 20;
            });
            el('circle', { cx: lx + 5, cy: ly - 4, r: 4, fill: COLOR.surface, stroke: COLOR.labs.other, 'stroke-width': 2 }, svg);
            label(svg, 'No reasoning', lx + 16, ly, { fill: COLOR.ink2, size: 13 });
        }

        /* Rating grid */
        var grid = el('g', {}, svg);
        for (var r = Math.ceil(rDomain[0] / 500) * 500; r <= rDomain[1]; r += 500) {
            el('line', { x1: plot.x0, x2: plot.x1, y1: y(r), y2: y(r), stroke: COLOR.grid, 'stroke-width': 1 }, grid);
            label(grid, formatRating(r), plot.x0 - 8, y(r) + 4, { anchor: 'end', tabular: true });
        }

        /* Time axis: a tick per half-year, the year named at each January. */
        el('line', { x1: plot.x0, x2: plot.x1, y1: plot.y1, y2: plot.y1, stroke: COLOR.axis, 'stroke-width': 1 }, grid);
        var firstYear = new Date(tDomain[0] * 1000).getUTCFullYear();
        var lastYear = new Date(tDomain[1] * 1000).getUTCFullYear();
        var yearPx = (plot.x1 - plot.x0) / ((tDomain[1] - tDomain[0]) / (365.25 * DAY));
        for (var year = firstYear; year <= lastYear + 1; year++) {
            [0, 6].forEach(function (month) {
                var ts = Date.UTC(year, month, 1) / 1000;
                if (ts < tDomain[0] || ts > tDomain[1]) return;
                var major = month === 0;
                el('line', {
                    x1: x(ts), x2: x(ts), y1: plot.y1, y2: plot.y1 + (major ? 6 : 4),
                    stroke: COLOR.axis, 'stroke-width': 1
                }, grid);
                if (major) {
                    label(grid, String(year), x(ts), plot.y1 + 22, { anchor: 'middle', fill: COLOR.ink2, tabular: true });
                } else if (yearPx > 150) {
                    label(grid, 'Jul', x(ts), plot.y1 + 22, { anchor: 'middle', size: 11 });
                }
            });
        }

        /* Skill bands: a second reading of the same axis, kept out of the plot. */
        if (!compact) {
            var tiers = data.tiers, gx = plot.x1 + 12;
            el('line', {
                x1: gx, x2: gx, y1: y(tiers[0].from), y2: plot.y0,
                stroke: COLOR.axis, 'stroke-width': 1
            }, grid);
            tiers.forEach(function (tier, i) {
                var upper = i + 1 < tiers.length ? tiers[i + 1].from : rDomain[1];
                el('line', {
                    x1: gx, x2: gx + 5, y1: y(tier.from), y2: y(tier.from),
                    stroke: COLOR.axis, 'stroke-width': 1
                }, grid);
                label(grid, tier.name, gx + 10, (y(tier.from) + y(upper)) / 2 + 4, { size: 11 });
            });
        }

        /* Lab staircase (only while a lab is focused) sits under the record line. */
        var labLine = el('path', {
            d: '', fill: 'none', 'stroke-width': 1.5, 'stroke-linejoin': 'round', opacity: 0
        }, svg);

        var endX = x(data.now);
        var recordPoints = points.filter(function (p) { return p.record; });
        var recordLine = el('path', {
            d: staircase(recordPoints, x, y, endX),
            fill: 'none', stroke: COLOR.record, 'stroke-width': 2,
            'stroke-linejoin': 'round', 'stroke-linecap': 'round'
        }, svg);

        /* Points: gray first, featured labs above, record-setters last. */
        var order = { other: 0, anthropic: 1, openai: 2, google: 3 };
        var drawn = points.map(function (p) {
            return { p: p, cx: x(p.ts), cy: y(p.rating), r: 5.5 };
        }).sort(function (a, b) {
            return (a.p.record - b.p.record) || (order[a.p.lab] - order[b.p.lab]);
        });
        var pointLayer = el('g', {}, svg);
        drawn.forEach(function (d) {
            var color = COLOR.labs[d.p.lab];
            var group = el('g', {}, pointLayer);
            // Record-setters sit on a light disc, which reads as a ring around the lab color.
            if (d.p.record) el('circle', { cx: d.cx, cy: d.cy, r: d.r + 3, fill: COLOR.record }, group);
            var attrs = { cx: d.cx, cy: d.cy };
            if (d.p.reasoning) {
                attrs.r = d.r;
                attrs.fill = color;
                if (!d.p.record) { attrs.stroke = COLOR.surface; attrs['stroke-width'] = 1.5; }
            } else {
                attrs.r = d.r - 1;
                attrs.fill = COLOR.surface;
                attrs.stroke = color;
                attrs['stroke-width'] = 2;
            }
            el('circle', attrs, group);
            d.node = group;
        });

        /* Direct labels for the record-setters only; newest gets first pick of position. */
        var labelLayer = el('g', {}, svg);
        var placed = [];
        var toLabel = compact ? records.slice(-1) : records.slice().reverse();
        var recordXY = records.map(function (rec) { return { x: x(rec.ts), y: y(rec.rating) }; });
        function lineHits(box) {
            for (var i = 0; i < recordXY.length; i++) {
                var a = recordXY[i], nextX = i + 1 < recordXY.length ? recordXY[i + 1].x : endX;
                // horizontal run at a.y from a.x to nextX
                if (a.y > box.y0 && a.y < box.y1 && a.x < box.x1 && nextX > box.x0) return true;
                // riser into this record
                if (i > 0 && a.x > box.x0 && a.x < box.x1 &&
                    Math.min(a.y, recordXY[i - 1].y) < box.y1 && Math.max(a.y, recordXY[i - 1].y) > box.y0) return true;
            }
            return false;
        }
        toLabel.forEach(function (rec) {
            var px = x(rec.ts), py = y(rec.rating);
            var sub = formatRating(rec.rating) + ' in ' + rec.month;
            var w = Math.max(textWidth(rec.id, 12, 600), textWidth(sub, 11)), h = 31;
            var candidates = [
                { x0: px - 13 - w, y0: py - 10 - h, anchor: 'end' },
                { x0: px + 13, y0: py - 10 - h, anchor: 'start' },
                { x0: px + 13, y0: py + 10, anchor: 'start' },
                { x0: px - 13 - w, y0: py + 10, anchor: 'end' }
            ];
            var best = null;
            candidates.forEach(function (c, index) {
                var box = { x0: c.x0 - 3, x1: c.x0 + w + 3, y0: c.y0 - 2, y1: c.y0 + h + 2 };
                if (box.x0 < plot.x0 || box.x1 > plot.x1 || box.y0 < plot.y0 - 28 || box.y1 > plot.y1) return;
                if (placed.some(function (other) { return overlaps(box, other); })) return;
                var cost = index * 0.5 + (lineHits(box) ? 6 : 0);
                drawn.forEach(function (d) {
                    if (d.cx + d.r > box.x0 && d.cx - d.r < box.x1 && d.cy + d.r > box.y0 && d.cy - d.r < box.y1) cost += 3;
                });
                if (!best || cost < best.cost) best = { cost: cost, c: c, box: box };
            });
            if (!best || best.cost >= 6) return;
            placed.push(best.box);
            var tx = best.c.anchor === 'end' ? best.c.x0 + w : best.c.x0;
            label(labelLayer, rec.id, tx, best.c.y0 + 12, { fill: COLOR.ink, size: 12, weight: 600, anchor: best.c.anchor });
            label(labelLayer, sub, tx, best.c.y0 + 27, { fill: COLOR.ink2, size: 11, anchor: best.c.anchor, tabular: true });
        });

        /* Hover marks, drawn above everything. */
        var hover = el('g', { 'pointer-events': 'none', opacity: 0 }, svg);
        var whisker = el('path', { d: '', fill: 'none', stroke: COLOR.ink2, 'stroke-width': 1.5 }, hover);
        var ring = el('circle', { r: 12, fill: 'none', stroke: COLOR.ink, 'stroke-width': 1.5 }, hover);

        return {
            svg: svg, plot: plot, x: x, y: y, endX: endX, drawn: drawn, width: width,
            labLine: labLine, recordLine: recordLine, labelLayer: labelLayer,
            hover: hover, whisker: whisker, ring: ring
        };
    }

    /* ---- Lab focus ---- */
    function activeLab() { return state.previewLab || state.focusLab; }

    function applyFocus() {
        var g = state.geometry, lab = activeLab();
        if (!g) return;
        g.drawn.forEach(function (d) {
            d.node.setAttribute('opacity', !lab || d.p.lab === lab ? 1 : 0.13);
        });
        g.recordLine.setAttribute('opacity', lab ? 0.3 : 1);
        g.labelLayer.setAttribute('opacity', lab ? 0.3 : 1);
        if (lab) {
            var own = data.points.filter(function (p) { return p.lab === lab; });
            g.labLine.setAttribute('d', staircase(own, g.x, g.y, g.endX));
            g.labLine.setAttribute('stroke', COLOR.labs[lab]);
            g.labLine.setAttribute('opacity', 1);
        } else {
            g.labLine.setAttribute('opacity', 0);
        }
        legendButtons.forEach(function (button) {
            var on = state.focusLab === button.dataset.lab;
            button.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
    }

    legendButtons.forEach(function (button) {
        var lab = button.dataset.lab;
        button.addEventListener('click', function () {
            state.focusLab = state.focusLab === lab ? null : lab;
            state.previewLab = null;
            applyFocus();
        });
        button.addEventListener('pointerenter', function (event) {
            if (event.pointerType === 'touch') return;
            state.previewLab = lab;
            applyFocus();
        });
        button.addEventListener('pointerleave', function () {
            state.previewLab = null;
            applyFocus();
        });
    });

    /* ---- Tooltip ---- */
    function row(name, value) {
        var line = document.createElement('div');
        line.className = 'timeline-tip-row';
        var k = document.createElement('span');
        k.textContent = name;
        var v = document.createElement('strong');
        v.textContent = value;
        line.appendChild(k);
        line.appendChild(v);
        return line;
    }

    function showPoint(d) {
        var g = state.geometry, p = d.p;
        g.ring.setAttribute('cx', d.cx);
        g.ring.setAttribute('cy', d.cy);
        if (p.low != null && p.high != null) {
            var top = g.y(p.high), bottom = g.y(p.low);
            g.whisker.setAttribute('d',
                'M' + (d.cx - 4) + ' ' + top + 'H' + (d.cx + 4) +
                'M' + d.cx + ' ' + top + 'V' + bottom +
                'M' + (d.cx - 4) + ' ' + bottom + 'H' + (d.cx + 4));
        } else {
            g.whisker.setAttribute('d', '');
        }
        g.hover.setAttribute('opacity', 1);

        tooltip.textContent = '';
        var title = document.createElement('div');
        title.className = 'timeline-tip-title';
        var swatch = document.createElement('i');
        swatch.style.background = p.reasoning ? COLOR.labs[p.lab] : 'transparent';
        swatch.style.borderColor = COLOR.labs[p.lab];
        title.appendChild(swatch);
        title.appendChild(document.createTextNode(p.id));
        tooltip.appendChild(title);
        tooltip.appendChild(row('Rating', formatRating(p.rating)));
        if (p.low != null && p.high != null) {
            tooltip.appendChild(row('95% range', formatRating(p.low) + ' to ' + formatRating(p.high)));
        }
        tooltip.appendChild(row('Lab', p.lab_name));
        tooltip.appendChild(row('Reasoning', p.reasoning ? 'Yes' : 'No'));
        tooltip.appendChild(row('Released', formatDate(p.ts, true)));
        if (p.games != null) tooltip.appendChild(row('Games', String(p.games)));
        if (p.record) {
            var note = document.createElement('div');
            note.className = 'timeline-tip-note';
            note.textContent = 'Set a new record on release';
            tooltip.appendChild(note);
        }

        tooltip.hidden = false;
        var box = tooltip.getBoundingClientRect();
        var left = d.cx + 16;
        if (left + box.width > g.width - 4) left = d.cx - 16 - box.width;
        var topPx = Math.max(4, Math.min(d.cy - box.height / 2, g.svg.clientHeight - box.height - 4));
        tooltip.style.left = Math.max(4, left) + 'px';
        tooltip.style.top = topPx + 'px';
    }

    function hidePoint() {
        if (state.geometry) state.geometry.hover.setAttribute('opacity', 0);
        tooltip.hidden = true;
    }

    function nearest(event) {
        var g = state.geometry;
        var bounds = g.svg.getBoundingClientRect();
        var mx = event.clientX - bounds.left, my = event.clientY - bounds.top;
        var lab = activeLab(), best = null, bestDist = 30 * 30;
        g.drawn.forEach(function (d) {
            if (lab && d.p.lab !== lab) return;
            var dist = (d.cx - mx) * (d.cx - mx) + (d.cy - my) * (d.cy - my);
            if (dist < bestDist) { bestDist = dist; best = d; }
        });
        return best;
    }

    function onPointer(event) {
        var hit = nearest(event);
        if (hit) showPoint(hit); else hidePoint();
    }

    /* ---- Mount, resize ---- */
    function render() {
        var width = Math.floor(mount.clientWidth);
        if (!width || (state.geometry && state.geometry.width === width)) return;
        var g = build(width, false);
        if (state.geometry) mount.removeChild(state.geometry.svg);
        mount.insertBefore(g.svg, mount.firstChild);
        g.svg.addEventListener('pointermove', onPointer);
        g.svg.addEventListener('pointerdown', onPointer);
        g.svg.addEventListener('pointerleave', hidePoint);
        state.geometry = g;
        hidePoint();
        applyFocus();
    }

    if (!data.points.length) return;
    render();
    if (window.ResizeObserver) new ResizeObserver(render).observe(mount);
    else window.addEventListener('resize', render);

    /* ---- PNG export ---- */
    var downloadBtn = document.getElementById('download-png');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function () {
            var svg = build(1400, true).svg;
            var source = new XMLSerializer().serializeToString(svg);
            var image = new Image();
            image.onload = function () {
                var canvas = document.createElement('canvas');
                canvas.width = 2800;
                canvas.height = 1400;
                var ctx = canvas.getContext('2d');
                ctx.scale(2, 2);
                ctx.drawImage(image, 0, 0);
                canvas.toBlob(function (blob) {
                    var link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = 'llm-chess-timeline.png';
                    link.click();
                    setTimeout(function () { URL.revokeObjectURL(link.href); }, 1000);
                }, 'image/png');
            };
            image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(source);
        });
    }
})();
