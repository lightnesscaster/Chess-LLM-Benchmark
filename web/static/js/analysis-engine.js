/* UCI output has no request IDs. Finish each search before starting its successor. */
(function (root) {
    'use strict';
    class AnalysisEngine {
        constructor(worker, onInfo, onDone = () => {}) {
            this.worker = worker;
            this.onInfo = onInfo;
            this.onDone = onDone;
            this.ready = false;
            this.active = null;
            this.target = null;
            this.stopping = false;
        }
        request(target) {
            this.target = target;
            if (this.active && this.active.key !== target.key && !this.stopping) {
                this.stopping = true;
                this.worker.postMessage('stop');
            }
            this.start();
        }
        start() {
            if (!this.ready || this.active || !this.target) return;
            this.active = this.target;
            this.worker.postMessage(this.active.command);
            this.worker.postMessage('go infinite');
        }
        stop() {
            this.target = null;
            if (this.active && !this.stopping) {
                this.stopping = true;
                this.worker.postMessage('stop');
            }
        }
        receive(line) {
            if (line === 'readyok') {
                this.ready = true;
                this.start();
                return;
            }
            if (line.startsWith('bestmove')) {
                const completed = this.active;
                const interrupted = this.stopping;
                this.active = null;
                this.stopping = false;
                if (!interrupted && completed?.key === this.target?.key) {
                    this.target = null;
                    this.onDone(completed);
                }
                this.start();
                return;
            }
            if (!this.active || this.stopping || this.target?.key !== this.active.key) return;
            if (/\b(?:lowerbound|upperbound)\b/.test(line)) return;
            const depth = line.match(/\bdepth (\d+)/);
            const score = line.match(/\bscore (cp|mate) (-?\d+)/);
            const pv = line.match(/\bpv (.+)/);
            if (!line.startsWith('info ') || !depth || !score || !pv) return;
            const index = Number(line.match(/\bmultipv (\d+)/)?.[1] || 1);
            this.onInfo({target: this.active, depth: Number(depth[1]), index,
                cp: score[1] === 'cp' ? Number(score[2]) : null,
                mate: score[1] === 'mate' ? Number(score[2]) : null,
                moves: pv[1].trim().split(/\s+/)});
        }
    }
    if (typeof module !== 'undefined') module.exports = {AnalysisEngine};
    else root.AnalysisEngine = AnalysisEngine;
})(typeof window !== 'undefined' ? window : globalThis);
