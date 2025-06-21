// tailwind.config.js - Tailwind CSS Configuration (Fixed)

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        // Custom brand colors
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        // Intelligence-specific colors
        threat: {
          low: '#10b981',
          medium: '#f59e0b',
          high: '#ef4444',
        },
        security: {
          classified: '#dc2626',
          restricted: '#ea580c',
          confidential: '#f59e0b',
          unclassified: '#10b981',
        }
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          '"Helvetica Neue"',
          'Arial',
          'sans-serif',
        ],
        mono: [
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          '"Liberation Mono"',
          '"Courier New"',
          'monospace',
        ],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      borderRadius: {
        '4xl': '2rem',
        '5xl': '2.5rem',
      },
      boxShadow: {
        'outline-blue': '0 0 0 3px rgba(59, 130, 246, 0.5)',
        'outline-red': '0 0 0 3px rgba(239, 68, 68, 0.5)',
        'outline-green': '0 0 0 3px rgba(16, 185, 129, 0.5)',
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
      },
      backdropBlur: {
        xs: '2px',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'fade-in-up': 'fadeInUp 0.5s ease-out',
        'slide-in-left': 'slideInLeft 0.5s ease-out',
        'slide-in-right': 'slideInRight 0.5s ease-out',
        'scale-in': 'scaleIn 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': {
            opacity: '0',
            transform: 'translateY(30px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        slideInLeft: {
          '0%': {
            transform: 'translateX(-100%)',
            opacity: '0',
          },
          '100%': {
            transform: 'translateX(0)',
            opacity: '1',
          },
        },
        slideInRight: {
          '0%': {
            transform: 'translateX(100%)',
            opacity: '0',
          },
          '100%': {
            transform: 'translateX(0)',
            opacity: '1',
          },
        },
        scaleIn: {
          '0%': {
            transform: 'scale(0.9)',
            opacity: '0',
          },
          '100%': {
            transform: 'scale(1)',
            opacity: '1',
          },
        },
      },
      screens: {
        'xs': '475px',
        '3xl': '1600px',
      },
      maxWidth: {
        '8xl': '88rem',
        '9xl': '96rem',
      },
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
      gridTemplateColumns: {
        'auto-fit': 'repeat(auto-fit, minmax(250px, 1fr))',
        'auto-fill': 'repeat(auto-fill, minmax(250px, 1fr))',
      },
    },
  },
  plugins: [
    // Custom plugin for intelligence-specific utilities
    function({ addUtilities, addComponents, theme }) {
      const newUtilities = {
        '.text-shadow': {
          'text-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
        },
        '.text-shadow-md': {
          'text-shadow': '0 4px 8px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.08)',
        },
        '.text-shadow-lg': {
          'text-shadow': '0 15px 30px rgba(0, 0, 0, 0.11), 0 5px 15px rgba(0, 0, 0, 0.08)',
        },
        '.text-shadow-none': {
          'text-shadow': 'none',
        },
        // Gradient text utilities
        '.gradient-text-blue': {
          'background': 'linear-gradient(135deg, #3b82f6, #1e40af)',
          '-webkit-background-clip': 'text',
          '-webkit-text-fill-color': 'transparent',
          'background-clip': 'text',
        },
        '.gradient-text-purple': {
          'background': 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
          '-webkit-background-clip': 'text',
          '-webkit-text-fill-color': 'transparent',
          'background-clip': 'text',
        },
        // Glass morphism utilities
        '.glass': {
          'background': 'rgba(255, 255, 255, 0.25)',
          'backdrop-filter': 'blur(10px)',
          'border': '1px solid rgba(255, 255, 255, 0.18)',
        },
        '.glass-dark': {
          'background': 'rgba(0, 0, 0, 0.25)',
          'backdrop-filter': 'blur(10px)',
          'border': '1px solid rgba(255, 255, 255, 0.18)',
        },
        // Scrollbar utilities
        '.scrollbar-thin': {
          'scrollbar-width': 'thin',
        },
        '.scrollbar-none': {
          'scrollbar-width': 'none',
          '&::-webkit-scrollbar': {
            'display': 'none',
          },
        },
      };

      const newComponents = {
        // Button components
        '.btn': {
          'padding': '0.5rem 1rem',
          'border-radius': '0.5rem',
          'font-weight': '500',
          'transition': 'all 0.2s',
          'display': 'inline-flex',
          'align-items': 'center',
          'justify-content': 'center',
          'border': 'none',
          'cursor': 'pointer',
          '&:disabled': {
            'opacity': '0.6',
            'cursor': 'not-allowed',
          },
        },
        '.btn-primary': {
          'background-color': theme('colors.blue.600'),
          'color': theme('colors.white'),
          '&:hover': {
            'background-color': theme('colors.blue.700'),
          },
          '&:focus': {
            'box-shadow': theme('boxShadow.outline-blue'),
          },
        },
        '.btn-secondary': {
          'background-color': theme('colors.gray.200'),
          'color': theme('colors.gray.800'),
          '&:hover': {
            'background-color': theme('colors.gray.300'),
          },
        },
        '.btn-danger': {
          'background-color': theme('colors.red.600'),
          'color': theme('colors.white'),
          '&:hover': {
            'background-color': theme('colors.red.700'),
          },
          '&:focus': {
            'box-shadow': theme('boxShadow.outline-red'),
          },
        },
        '.btn-success': {
          'background-color': theme('colors.green.600'),
          'color': theme('colors.white'),
          '&:hover': {
            'background-color': theme('colors.green.700'),
          },
          '&:focus': {
            'box-shadow': theme('boxShadow.outline-green'),
          },
        },
        // Card components
        '.card': {
          'background-color': theme('colors.white'),
          'border': `1px solid ${theme('colors.gray.200')}`,
          'border-radius': theme('borderRadius.lg'),
          'box-shadow': theme('boxShadow.sm'),
        },
        '.card-header': {
          'padding': '1.5rem 1.5rem 1rem 1.5rem',
          'border-bottom': `1px solid ${theme('colors.gray.200')}`,
        },
        '.card-body': {
          'padding': '1.5rem',
        },
        '.card-footer': {
          'padding': '1rem 1.5rem 1.5rem 1.5rem',
          'border-top': `1px solid ${theme('colors.gray.200')}`,
          'background-color': theme('colors.gray.50'),
        },
        // Alert components
        '.alert': {
          'padding': '1rem',
          'border-radius': theme('borderRadius.lg'),
          'border': '1px solid',
        },
        '.alert-info': {
          'background-color': theme('colors.blue.50'),
          'border-color': theme('colors.blue.200'),
          'color': theme('colors.blue.800'),
        },
        '.alert-success': {
          'background-color': theme('colors.green.50'),
          'border-color': theme('colors.green.200'),
          'color': theme('colors.green.800'),
        },
        '.alert-warning': {
          'background-color': theme('colors.yellow.50'),
          'border-color': theme('colors.yellow.200'),
          'color': theme('colors.yellow.800'),
        },
        '.alert-error': {
          'background-color': theme('colors.red.50'),
          'border-color': theme('colors.red.200'),
          'color': theme('colors.red.800'),
        },
        // Badge components
        '.badge': {
          'display': 'inline-flex',
          'align-items': 'center',
          'padding': '0.25rem 0.625rem',
          'border-radius': theme('borderRadius.full'),
          'font-size': theme('fontSize.xs'),
          'font-weight': theme('fontWeight.medium'),
        },
        '.badge-blue': {
          'background-color': theme('colors.blue.100'),
          'color': theme('colors.blue.800'),
        },
        '.badge-green': {
          'background-color': theme('colors.green.100'),
          'color': theme('colors.green.800'),
        },
        '.badge-yellow': {
          'background-color': theme('colors.yellow.100'),
          'color': theme('colors.yellow.800'),
        },
        '.badge-red': {
          'background-color': theme('colors.red.100'),
          'color': theme('colors.red.800'),
        },
        '.badge-gray': {
          'background-color': theme('colors.gray.100'),
          'color': theme('colors.gray.800'),
        },
      };

      addUtilities(newUtilities);
      addComponents(newComponents);
    },
  ],
}