'use client'

import { useTheme } from 'next-themes'
import { type JSX, useEffect, useState } from 'react'

export default function ThemeToggle(): JSX.Element {
  // Track whether the component has mounted to avoid hydration mismatches
  const [mounted, setMounted] = useState<boolean>(false)

  // `resolvedTheme` can be `"light" | "dark" | undefined` depending on the current theme
  // `setTheme` expects a string corresponding to a valid theme value
  const { systemTheme, theme, setTheme } = useTheme();

  const currentTheme = theme === 'system' ? systemTheme : theme;

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return <div className="w-8 h-8" /> // Prevents layout shift
  }

  return (
    <button
      onClick={() => setTheme(currentTheme === 'dark' ? 'light' : 'dark')}
      className="w-8 h-8 flex items-center justify-center rounded-md bg-gray-100 dark:bg-gray-800 transition-colors hover:bg-gray-200 dark:hover:bg-gray-700"
      aria-label={currentTheme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
    >
      <span className="text-lg" aria-hidden="true">
        {currentTheme === 'dark' ? '🌞' : '🌙'}
      </span>
    </button>
  )
}