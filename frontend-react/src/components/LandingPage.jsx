import React, { useEffect, useRef } from 'react';
import { ChevronRight, Cpu, Database, Lock } from 'lucide-react';
import Button from './ui/Button';
import Card from './ui/Card';

const ParticleBackground = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId;
    let particles = [];

    // Configuration
    const particleColor = 'rgba(99, 102, 241, 0.3)'; // Indigo
    const lineColor = 'rgba(99, 102, 241, 0.15)';
    const connectionDistance = 120;

    const createParticle = () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2 + 1,
    });

    const initParticles = () => {
      // Density calculation: 1 particle per 9000px sq
      const particleCount = Math.floor((canvas.width * canvas.height) / 9000);
      particles = Array.from({ length: particleCount }, createParticle);
    };

    const resizeCanvas = () => {
      if (!canvas.parentElement) return;
      canvas.width = canvas.parentElement.offsetWidth;
      canvas.height = canvas.parentElement.offsetHeight;
      initParticles();
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Update and draw particles
      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;

        // Boundary check (bounce)
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        // Draw particle
        ctx.fillStyle = particleColor;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();

        // Connect particles
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < connectionDistance) {
            ctx.beginPath();
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 1 - distance / connectionDistance;
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      });

      animationFrameId = requestAnimationFrame(draw);
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    draw();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />;
};

const LandingPage = ({ onGetStarted }) => {
  useEffect(() => {
    const elements = Array.from(document.querySelectorAll('.scroll-fade-in'));
    if (!elements.length) return undefined;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: '0px 0px -10% 0px' }
    );

    elements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  return (
    <div className="flex-1 flex flex-col">
      <section className="relative py-20 lg:py-32 overflow-hidden scroll-fade-in">
        <ParticleBackground />
        <div className="container px-4 md:px-6 mx-auto relative z-10">
          <div className="flex flex-col items-center text-center max-w-4xl mx-auto space-y-8">
            <div className="inline-flex items-center rounded-full border border-indigo-500/30 bg-indigo-500/10 px-3 py-1 text-sm font-medium text-indigo-300 backdrop-blur-xl">
              <span className="flex h-2 w-2 rounded-full bg-indigo-500 mr-2 animate-pulse"></span>
              v2.0 Now Available with H5AD Support
            </div>

            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white">
              Precision{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">
                Glioma Analysis
              </span>{' '}
              for Modern Research
            </h1>

            <p className="text-xl text-slate-400 max-w-2xl leading-relaxed">
              Leverage advanced vector embeddings and single-cell RNA sequencing to annotate
              brain tumor data with unprecedented accuracy.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 w-full justify-center pt-4">
              <Button onClick={onGetStarted} className="px-8 py-4 text-lg group">
                Start Analyzing{' '}
                <ChevronRight className="ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
              <Button variant="secondary" className="px-8 py-4 text-lg">
                View Demo Data
              </Button>
            </div>
          </div>
        </div>

        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-indigo-600/20 rounded-full blur-[120px] -z-10" />
        <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-cyan-600/10 rounded-full blur-[100px] -z-10" />
      </section>

      <section className="py-20 bg-slate-900/50 border-t border-slate-800 scroll-fade-in">
        <div className="container px-4 mx-auto">
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Cpu,
                title: 'Vector Embeddings',
                desc: 'High-dimensional space mapping for precise cell clustering.',
              },
              {
                icon: Database,
                title: 'H5AD Support',
                desc: 'Native support for standard AnnData formats used in scRNA-seq.',
              },
              {
                icon: Lock,
                title: 'Secure Processing',
                desc: 'Local-first processing pipeline ensuring data privacy.',
              },
            ].map((feature, index) => (
              <Card
                key={index}
                className="hover:bg-slate-800 transition-colors cursor-default"
              >
                <div className="w-12 h-12 bg-slate-700/50 rounded-lg flex items-center justify-center mb-4">
                  <feature.icon className="text-indigo-400" size={24} />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                <p className="text-slate-400">{feature.desc}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;

