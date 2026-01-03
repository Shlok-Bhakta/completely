import * as vscode from 'vscode';

const API_URL = 'http://localhost:8000/v1/completions';

let out: vscode.OutputChannel;

async function getCompletion(prefix: string, suffix: string, token: vscode.CancellationToken): Promise<string> {
	out.appendLine(`[REQ] prefix=${prefix.length} suffix=${suffix.length}`);
	
	const controller = new AbortController();
	const dispose = token.onCancellationRequested(() => controller.abort());
	
	try {
		const res = await fetch(API_URL, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ prompt: prefix.slice(-1000), suffix: suffix.slice(0, 500), max_tokens: 50 }),
			signal: controller.signal,
		});
		const data = await res.json() as { choices: { text: string }[] };
		out.appendLine(`[RES] ${JSON.stringify(data)}`);
		return data.choices?.[0]?.text ?? '';
	} finally {
		dispose.dispose();
	}
}

export function activate(context: vscode.ExtensionContext) {
	out = vscode.window.createOutputChannel('Completely');
	out.show();
	out.appendLine('Completely activated!');

	const provider: vscode.InlineCompletionItemProvider = {
		async provideInlineCompletionItems(document, position, context, token) {
			if (token.isCancellationRequested) {
				return { items: [] };
			}
			
			out.appendLine(`[TRIGGER] ${document.fileName}:${position.line}:${position.character}`);
			
			try {
				const prefix = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
				const suffix = document.getText(new vscode.Range(position, new vscode.Position(document.lineCount, 0)));
				
				const completion = await getCompletion(prefix, suffix, token);
				
				if (token.isCancellationRequested) {
					out.appendLine(`[CANCELLED]`);
					return { items: [] };
				}
				
				if (!completion || !completion.trim()) {
					out.appendLine(`[EMPTY]`);
					return { items: [] };
				}
				
				out.appendLine(`[COMPLETION] "${completion.slice(0, 80).replace(/\n/g, '\\n')}"`);
				
				const item = new vscode.InlineCompletionItem(completion);
				item.range = new vscode.Range(position, position);
				
				out.appendLine(`[RETURN] item created at ${position.line}:${position.character}`);
				return { items: [item] };
			} catch (e: any) {
				if (e.name === 'AbortError') {
					out.appendLine(`[ABORTED]`);
				} else {
					out.appendLine(`[ERROR] ${e}`);
				}
				return { items: [] };
			}
		}
	};

	context.subscriptions.push(
		vscode.languages.registerInlineCompletionItemProvider({ pattern: '**/*.py' }, provider)
	);
	out.appendLine('Provider registered for **/*.py');
}

export function deactivate() {}
