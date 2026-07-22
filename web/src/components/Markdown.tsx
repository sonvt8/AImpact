import DOMPurify from 'dompurify'
import { marked } from 'marked'

export default function Markdown({ text }: { text: string }) {
  const html = DOMPurify.sanitize(marked.parse(text, { breaks: true }) as string)
  return <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
}
